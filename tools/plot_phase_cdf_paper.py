#!/usr/bin/env python3
"""Export a saved-CDF phase function as a publication figure, without data repair.

The existing reader, FP64 reconstruction, validation, and GLOBAL color-limit
selection are reused from plot_phase_cdf. Only typography and output differ.
No tracing, smoothing, flooring, resampling of CDF values, or renormalization is
performed. Labels and tick marks remain text in PDF/SVG; the heatmap is an image.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Sequence
import warnings

import numpy as np

from plot_phase_cdf import Backend, PlotData, open_record, plot_limits, reconstruct


@dataclass(frozen=True)
class _FontInfo:
    path: Path
    family: str
    names: frozenset[str]
    italic: bool
    weight: int
    truetype: bool
    variable: bool


@dataclass(frozen=True)
class ResolvedFonts:
    report: dict[str, Any]


class _MathFontWarnings(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # MathText reports missing symbols through logging rather than Python's
        # warnings module. Do not accept a substituted dummy glyph in a label.
        raise ValueError("MathText font warning: " + record.getMessage())


def _font_info(path: Path) -> _FontInfo:
    from fontTools.ttLib import TTFont
    from matplotlib.ft2font import FT2Font

    with TTFont(str(path), fontNumber=0, lazy=True) as font:
        names = frozenset(n.toUnicode().casefold() for n in font["name"].names
                          if n.nameID in (1, 16))
        weight = int(font["OS/2"].usWeightClass) if "OS/2" in font else 400
        truetype, variable = "glyf" in font, "fvar" in font
    face = FT2Font(str(path))
    flags = getattr(face.style_flags, "value", face.style_flags)
    return _FontInfo(path.resolve(), face.family_name, names,
                     bool(int(flags) & 1), weight, truetype, variable)


def _clear_font_cache() -> None:
    from matplotlib import font_manager

    # Matplotlib has addfont but no public removefont. This clears entries that
    # referred to our temporary variable-font instance when its lifetime ends.
    font_manager.fontManager._findfont_cached.cache_clear()


def _release_rendered_fonts() -> None:
    from matplotlib import font_manager
    from matplotlib.mathtext import MathTextParser

    # FreeType retains file streams. PDF/SVG MathText results also retain faces,
    # and cached bound methods inside a fontset can form reference cycles. Clear
    # those references before removing a temporary TTF, especially on Windows.
    font_manager._get_font.cache_clear()
    MathTextParser._parse_cached.cache_clear()
    gc.collect()


def _register_directory(directory: Path) -> None:
    from matplotlib import font_manager

    if not directory.is_dir():
        raise ValueError(f"Font directory does not exist: {directory}")
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".ttf", ".otf", ".ttc"):
            font_manager.fontManager.addfont(str(path))


def _resolve_font(family: str, *, italic: bool, require_truetype: bool,
                  explicit: Path | None = None) -> _FontInfo:
    from matplotlib import font_manager

    aliases = {family.casefold()}
    if family.casefold() in {"linux libertine", "linux libertine o"}:
        aliases.update(("linux libertine", "linux libertine o"))

    def candidate_paths() -> list[Path]:
        if explicit is not None:
            if not explicit.is_file():
                raise ValueError(f"Font file does not exist: {explicit}")
            return [explicit]
        # A variable Noto Sans JP file can be listed as 'Noto Sans JP Thin'; its
        # typographic family (name ID 16) is checked before it is accepted below.
        return sorted({Path(entry.fname) for entry in font_manager.fontManager.ttflist
                       if any(entry.name.casefold() == name
                              or entry.name.casefold().startswith(name + " ")
                              for name in aliases) and Path(entry.fname).is_file()})

    paths = candidate_paths()
    if not paths and explicit is None:
        # Refresh system discovery once; an existing Matplotlib cache may predate
        # a font installation. --jp-font-file also handles Windows per-user fonts.
        known = {str(Path(entry.fname).resolve()) for entry in font_manager.fontManager.ttflist}
        for path in font_manager.findSystemFonts():
            if str(Path(path).resolve()) not in known:
                font_manager.fontManager.addfont(path)
        paths = candidate_paths()
    candidates = [_font_info(path) for path in paths]
    candidates = [font for font in candidates if font.names.intersection(aliases)
                  and font.italic == italic and (font.variable or 300 <= font.weight <= 500)]
    matching_style = bool(candidates)
    if require_truetype:
        candidates = [font for font in candidates if font.truetype]
    if not candidates:
        style = "italic" if italic else "regular"
        detail = ("Only CFF/OpenType outlines were found; use the TrueType (TTF/glyf) edition "
                  "for this PDF exporter. " if matching_style and require_truetype else "")
        raise ValueError(
            f"Required {style} font '{family}' was not found. {detail}"
            "Install that font, provide its directory with --font-dir, or use "
            "--jp-font-file for a Japanese font file. An intentional alternate Latin "
            "family can be selected with --latin-font; no fallback was performed."
        )
    return min(candidates, key=lambda font: (font.variable, abs(font.weight - 400), str(font.path)))


def _regular_instance(font: _FontInfo, directory: Path) -> _FontInfo:
    if not font.variable:
        return font
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    with TTFont(str(font.path), fontNumber=0) as variable:
        axes = {axis.axisTag: axis.defaultValue for axis in variable["fvar"].axes}
        if "wght" not in axes:
            raise ValueError(f"Variable font has no wght axis: {font.path}")
        axes["wght"] = 400
        # Resolve every axis to a constant, so FreeType cannot silently choose
        # Noto's Thin default instance. Other axes retain the font's defaults.
        static = instantiateVariableFont(variable, axes, inplace=False, updateFontNames=True)
        destination = directory / ("italic.ttf" if font.italic else "regular.ttf")
        static.save(str(destination))
        static.close()
    return _font_info(destination)


def _require_glyphs(font: _FontInfo, characters: str) -> None:
    from matplotlib.ft2font import FT2Font

    face = FT2Font(str(font.path))
    missing = sorted({char for char in characters if not char.isspace()
                      and face.get_char_index(ord(char)) == 0})
    if missing:
        values = ", ".join(f"{char!r} (U+{ord(char):04X})" for char in missing)
        raise ValueError(f"Font '{font.family}' lacks required glyphs: {values}; no fallback performed")


@contextmanager
def font_context(*, language: str = "ja", latin_font: str = "Linux Libertine",
                 jp_font: str = "Noto Sans JP", font_dirs: Sequence[Path] = (),
                 jp_font_file: Path | None = None,
                 require_truetype: bool = True) -> Iterator[ResolvedFonts]:
    """Resolve real fonts, including a regular instance of variable Noto Sans JP.

    Keep this context open until the figure has been saved. Font registrations
    and temporary font instances are removed on exit. No font is downloaded.
    """
    import matplotlib
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties

    if language not in ("ja", "en"):
        raise ValueError("language must be ja or en")
    original_entries = list(font_manager.fontManager.ttflist)
    with tempfile.TemporaryDirectory(prefix="rainbow-paper-fonts-") as temporary:
        try:
            bundled = Path(__file__).resolve().parent / "fonts" / "libertine"
            if latin_font.casefold() in {"linux libertine", "linux libertine o"} and bundled.is_dir():
                _register_directory(bundled)
            for directory in font_dirs:
                _register_directory(Path(directory))
            latin = _resolve_font(latin_font, italic=False, require_truetype=require_truetype)
            italic = _resolve_font(latin_font, italic=True, require_truetype=require_truetype)
            japanese = (_resolve_font(jp_font, italic=False, require_truetype=require_truetype,
                                     explicit=jp_font_file) if language == "ja" else None)
            sources = {"latin_regular": latin, "latin_italic": italic, "japanese_regular": japanese}
            # Separate directories avoid collisions if intentional alternate
            # Latin and Japanese families both happen to be variable fonts.
            selected: dict[str, _FontInfo | None] = {}
            for role, source in sources.items():
                directory = Path(temporary) / role
                directory.mkdir()
                selected[role] = None if source is None else _regular_instance(source, directory)
            latin, italic, japanese = (selected[role] for role in sources)
            assert latin is not None and italic is not None
            _require_glyphs(latin, "0123456789-−+().[]/degpsrloAzimuthScatteringan")
            _require_glyphs(italic, "pθφϕω")
            if japanese is not None:
                _require_glyphs(japanese, "方位角散乱")
            chosen = [font for font in selected.values() if font is not None]
            families = {font.family.casefold() for font in chosen}
            # A system CFF font and bundled TTF can have the same family name.
            # Exclude the competing entries for the lifetime of this render.
            font_manager.fontManager.ttflist[:] = [
                entry for entry in font_manager.fontManager.ttflist
                if entry.name.casefold() not in families]
            for font in chosen:
                font_manager.fontManager.addfont(str(font.path))
            _clear_font_cache()
            regular_pattern = FontProperties(fname=str(latin.path)).get_fontconfig_pattern()
            italic_pattern = FontProperties(fname=str(italic.path)).get_fontconfig_pattern()
            family_order = [latin.family] + ([] if japanese is None else [japanese.family])
            rc = {
                "font.family": family_order, "font.weight": "normal", "font.style": "normal",
                "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False,
                "svg.fonttype": "none", "text.usetex": False,
                "mathtext.fontset": "custom", "mathtext.fallback": None,
                "mathtext.rm": regular_pattern, "mathtext.it": italic_pattern,
                # These figures use no script/bold/typewriter symbols. Bind all
                # slots explicitly so Matplotlib never looks for a default font.
                "mathtext.cal": regular_pattern, "mathtext.sf": regular_pattern,
                "mathtext.tt": regular_pattern, "mathtext.bf": regular_pattern,
                "mathtext.bfit": italic_pattern,
                # Use this version's factory default, not a style-file override:
                # its semantics changed in 3.11. Variables are explicit below.
                "mathtext.default": matplotlib.rcParamsDefault["mathtext.default"],
                "figure.facecolor": "white", "axes.facecolor": "white",
                "savefig.facecolor": "white", "savefig.transparent": False,
                "savefig.bbox": None, "svg.image_inline": True,
                "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
                "xtick.color": "black", "ytick.color": "black",
            }
            report: dict[str, Any] = {"family_order": family_order}
            for role, source in sources.items():
                actual = selected[role]
                report[role] = (None if source is None or actual is None else {
                    "family": actual.family, "source_file": str(source.path),
                    "truetype_outlines": actual.truetype,
                    "variable_instantiated": source.variable,
                    "render_weight": actual.weight,
                })
            logger = logging.getLogger("matplotlib._mathtext")
            handler = _MathFontWarnings(level=logging.WARNING)
            logger.addHandler(handler)
            try:
                with matplotlib.rc_context(rc):
                    yield ResolvedFonts(report)
            finally:
                logger.removeHandler(handler)
        finally:
            font_manager.fontManager.ttflist[:] = original_entries
            _clear_font_cache()
            _release_rendered_fonts()


def _density_tick(value: float, _position: int | None = None) -> str:
    # The position IS log10(p); exponentiate only in the written notation.
    exponent = "0" if abs(value) < 1e-12 else f"{value:.6g}"
    return rf"$10^{{{exponent}}}$"


def _label_family(label: str) -> str:
    import matplotlib

    families = matplotlib.rcParams["font.family"]
    if any(ord(char) >= 0x2E80 for char in label):
        if len(families) < 2:
            raise ValueError("A Japanese label requires --language ja and its configured Japanese font")
        # In mixed text/MathText labels the non-math part uses one font, not
        # glyph-by-glyph family fallback. Latin units/variables must be in $...$
        # with explicit math font commands. Numeric axis ticks are separate text.
        return families[1]
    return families[0]


def create_figure(data: PlotData, *, log_limits: tuple[float, float], language: str = "ja",
                  width_mm: float = 170.0, height_mm: float = 95.0,
                  font_size: float = 8.0, dpi: int = 600, cmap: str = "viridis",
                  colorbar: str = "density", xlabel: str | None = None,
                  ylabel: str | None = None, colorbar_label: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Create a figure under font_context; caller owns closing/saving the figure."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    for name, value in (("width_mm", width_mm), ("height_mm", height_mm), ("font_size", font_size)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if type(dpi) is not int or dpi < 1:
        raise ValueError("DPI must be a positive integer")
    if language not in ("ja", "en") or colorbar not in ("density", "log10"):
        raise ValueError("Unsupported language or colorbar mode")
    lo, hi = map(float, log_limits)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError("Color limits must be finite and increasing")
    colors = matplotlib.colormaps[cmap].copy()
    colors.set_bad("white")
    values = np.ma.array(data.log10_pdf, mask=np.isneginf(data.log10_pdf), copy=False)
    norm = Normalize(vmin=lo, vmax=hi, clip=False)  # Data are ALREADY log10.
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4),
                           dpi=dpi, layout="constrained")
    try:
        fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.035, hspace=0.035)
        # Exact saved angular edges, including intersected cells at crop bounds.
        # No interpolation is applied to nonuniform cells or their log values.
        artist = ax.pcolormesh(data.phi_edges_deg, data.theta_edges_deg, values,
                               shading="flat", antialiased=False, edgecolors="none",
                               rasterized=True, norm=norm, cmap=colors)
        ax.set_xlim(*data.phi_range)
        ax.set_ylim(data.theta_range[1], data.theta_range[0])
        default_x = ("方位角 " if language == "ja" else "Azimuth ") + r"$\mathit{\phi}\;\mathrm{[deg]}$"
        default_y = ("散乱角 " if language == "ja" else "Scattering angle ") + r"$\mathit{\theta}\;\mathrm{[deg]}$"
        x_label, y_label = default_x if xlabel is None else xlabel, default_y if ylabel is None else ylabel
        ax.set_xlabel(x_label, fontsize=font_size, fontfamily=_label_family(x_label))
        ax.set_ylabel(y_label, fontsize=font_size, fontfamily=_label_family(y_label))
        ax.tick_params(axis="both", which="both", labelsize=font_size, width=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        visible = data.log10_pdf[np.isfinite(data.log10_pdf)]
        below, above = int(np.count_nonzero(visible < lo)), int(np.count_nonzero(visible > hi))
        extend = "both" if below and above else "min" if below else "max" if above else "neither"
        cb = fig.colorbar(artist, ax=ax, pad=0.025, fraction=0.055, aspect=25, extend=extend)
        if colorbar == "density":
            label = r"$\mathit{p}(\mathit{\omega})\;[\mathrm{sr}^{-1}]$"
            cb.formatter = FuncFormatter(_density_tick)
            cb.locator = MaxNLocator(nbins=7, integer=True, min_n_ticks=2)
        else:
            label = r"$\log_{10}(\mathit{p}(\mathit{\omega})/(1\,\mathrm{sr}^{-1}))$"
            cb.locator = MaxNLocator(nbins=6, min_n_ticks=3)
        cb.update_ticks()
        cb.minorticks_off()
        c_label = label if colorbar_label is None else colorbar_label
        cb.set_label(c_label, fontsize=font_size, fontfamily=_label_family(c_label))
        cb.ax.tick_params(labelsize=font_size, width=0.6)
        cb.outline.set_linewidth(0.6)
        if cb.solids is not None:
            cb.solids.set_rasterized(False)
            cb.solids.set_edgecolor("face")
        fig.canvas.draw()
        pixels = ax.get_window_extent().size
        summary = {
            "render_method": "pcolormesh_flat_no_antialias_rasterized",
            "figure_size_mm": [width_mm, height_mm], "font_size_pt": font_size,
            "raster_dpi": dpi, "axes_pixels_width_height": [float(value) for value in pixels],
            "language": language, "colormap": cmap,
            "color_normalization": "Normalize_on_existing_log10_pdf",
            "colorbar_mode": colorbar, "color_log10_limits": [lo, hi],
            "color_saturated_below_cells_in_view": below,
            "color_saturated_above_cells_in_view": above,
            "zero_cell_display": "white (masked exact zero); no positive PDF floor",
            "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
            "colorbar_label": cb.ax.get_ylabel(),
            "cdf_cells_exceed_render_pixels": bool(
                data.log10_pdf.shape[1] > pixels[0] or data.log10_pdf.shape[0] > pixels[1]),
            "heatmap_rasterized": True, "interpolation": "none",
        }
        return fig, summary
    except Exception:
        plt.close(fig)
        raise


def _output_path(path: Path, *, record_dir: Path, overwrite: bool,
                 suffixes: tuple[str, ...]) -> Path:
    resolved = path.resolve()
    if resolved.suffix.lower() not in suffixes:
        raise ValueError(f"Output suffix must be one of {suffixes}: {path}")
    if resolved.is_relative_to(record_dir.resolve()):
        raise ValueError("Write figure, preview, and report outside the dataset-record directory")
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}; specify --overwrite to replace it")
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"Output is not a regular file: {path}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", type=Path, help="saved record directory containing metadata.json and NPY arrays")
    parser.add_argument("--out", required=True, type=Path, help="PDF, SVG, or PNG outside the record directory")
    parser.add_argument("--preview", type=Path, help="optional additional PNG preview")
    parser.add_argument("--report", type=Path, help="JSON report (default: output path with .json suffix)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output, preview, and report")
    parser.add_argument("--language", choices=("ja", "en"), default="ja")
    parser.add_argument("--width-mm", type=float, default=170.0)
    parser.add_argument("--height-mm", type=float, default=95.0)
    parser.add_argument("--font-size", type=float, default=8.0, help="font size in points at the chosen final size")
    parser.add_argument("--dpi", type=int, default=600, help="raster heatmap resolution; vector text is unaffected")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--colorbar", choices=("density", "log10"), default="density")
    parser.add_argument("--xlabel", help="override axis label; Matplotlib $...$ math is supported")
    parser.add_argument("--ylabel", help="override axis label; Matplotlib $...$ math is supported")
    parser.add_argument("--colorbar-label", help="override colorbar label without changing data or normalization")
    parser.add_argument("--latin-font", default="Linux Libertine", help="Latin/math family (regular and italic required)")
    parser.add_argument("--jp-font", default="Noto Sans JP", help="Japanese family for --language ja")
    parser.add_argument("--font-dir", action="append", type=Path, default=[], help="register a font directory (repeatable)")
    parser.add_argument("--jp-font-file", type=Path, help="explicit Japanese TTF, including a variable Noto Sans JP TTF")
    parser.add_argument("--device", default="auto", help="auto (default), cpu, cuda, cuda:N")
    parser.add_argument("--chunk-phi", type=int, default=256)
    parser.add_argument("--theta-range", nargs=2, type=float, default=(0.0, 180.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--phi-range", nargs=2, type=float, default=(-180.0, 180.0), metavar=("LOW", "HIGH"))
    color = parser.add_mutually_exclusive_group()
    color.add_argument("--log-limits", nargs=2, type=float, metavar=("LOW", "HIGH"), help="fixed log10 color limits")
    color.add_argument("--decades", type=float, help="color range below GLOBAL maximum; no data changes")
    args = parser.parse_args(argv)
    fig = None
    temporary_outputs: list[Path] = []
    try:
        out = _output_path(args.out, record_dir=args.record, overwrite=args.overwrite,
                           suffixes=(".pdf", ".svg", ".png"))
        preview = None if args.preview is None else _output_path(
            args.preview, record_dir=args.record, overwrite=args.overwrite, suffixes=(".png",))
        report = _output_path(args.report or out.with_suffix(".json"), record_dir=args.record,
                              overwrite=args.overwrite, suffixes=(".json",))
        destinations = [path for path in (out, preview, report) if path is not None]
        if len(set(destinations)) != len(destinations):
            raise ValueError("Output, preview, and report paths must be distinct")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fail on absent required fonts before reading a potentially large CDF.
        with font_context(language=args.language, latin_font=args.latin_font, jp_font=args.jp_font,
                          font_dirs=args.font_dir, jp_font_file=args.jp_font_file,
                          require_truetype=(out.suffix.lower() == ".pdf")) as fonts:
            backend = Backend(args.device)
            print(f"[cdf-paper] {backend.name}: {backend.selection_note}")
            with open_record(args.record) as record:
                data = reconstruct(record, backend=backend, chunk_phi=args.chunk_phi,
                                   theta_range=tuple(args.theta_range), phi_range=tuple(args.phi_range))
            limits = plot_limits(data, None if args.log_limits is None else tuple(args.log_limits), args.decades)
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=r".*Glyph .*missing from font.*", category=UserWarning)
                warnings.filterwarnings("error", message=r"constrained_layout not applied.*", category=UserWarning)
                fig, drawing = create_figure(
                    data, log_limits=limits, language=args.language, width_mm=args.width_mm,
                    height_mm=args.height_mm, font_size=args.font_size, dpi=args.dpi, cmap=args.cmap,
                    colorbar=args.colorbar, xlabel=args.xlabel, ylabel=args.ylabel, colorbar_label=args.colorbar_label)
                summary = dict(data.summary)
                summary.update(drawing)
                summary.update({"schema": "rainbow.phase_cdf.paper_plot.v1", "source_record": str(args.record.resolve()),
                                "source_metadata": data.metadata, "output": str(out),
                                "preview": None if preview is None else str(preview), "report": str(report),
                                "fonts": fonts.report, "matplotlib_version": matplotlib.__version__})
                report_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                # Write all products successfully before exposing their final names.
                pairs: list[tuple[Path, Path]] = []
                for destination in destinations:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    descriptor, filename = tempfile.mkstemp(prefix=".cdf-paper-", suffix=destination.suffix,
                                                           dir=destination.parent)
                    os.close(descriptor)
                    temporary = Path(filename)
                    temporary_outputs.append(temporary)
                    pairs.append((temporary, destination))
                    if destination == report:
                        temporary.write_text(report_text, encoding="utf-8")
                    else:
                        fig.savefig(temporary, format=destination.suffix.lower()[1:], dpi=args.dpi,
                                    facecolor="white", transparent=False)  # Exact page size; never bbox_inches='tight'.
                for temporary, destination in pairs:
                    if destination.exists() and not args.overwrite:
                        raise FileExistsError(f"Output appeared during rendering: {destination}")
                    temporary.replace(destination)
            plt.close(fig)
            fig = None
        s = data.summary
        print(f"[cdf-paper] mass={s['cdf_reconstructed_mass']:.17g}, g={s['g_reconstructed']:.17g}, "
              f"metadata_g={s['g_metadata']}, zero_cells={s['zero_probability_cells']}, "
              f"positive_mass_product_underflows={s['positive_mass_product_underflows']}")
        print(f"[cdf-paper] global log10 PDF range=[{s['global_log10_pdf_min_positive']:.8g}, "
              f"{s['global_log10_pdf_max']:.8g}], color range={limits}")
        print("[cdf-paper] source_quality=" + json.dumps(s["source_quality"], ensure_ascii=False))
        print("[cdf-paper] storage_processing=" + json.dumps(s["storage_processing"], ensure_ascii=False))
        print(f"[cdf-paper] exact zero cells are white; {s['display_zero_cells']} in this view. "
              "No smoothing, CDF repair, PDF floor, or renormalization.")
        if drawing["cdf_cells_exceed_render_pixels"]:
            print("[cdf-paper] WARNING: more CDF cells than heatmap pixels; finite raster resolution can hide "
                  "detail. Use a crop, larger figure, or larger --dpi.", file=sys.stderr)
        print(f"[cdf-paper] saved {out}")
        if preview is not None:
            print(f"[cdf-paper] saved {preview}")
        print(f"[cdf-paper] saved {report}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError, UserWarning) as error:
        print(f"plot_phase_cdf_paper: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)
        for temporary in temporary_outputs:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
