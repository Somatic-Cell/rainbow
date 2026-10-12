"""Regression tests for paper exports, reusing the saved-CDF test fixtures.

Run with ``python -m unittest discover -s tests -p test_plot_phase_cdf_paper.py``.
PDF structure/text checks require the optional ``pypdf`` package. Set
``RAINBOW_TEST_JP_FONT_FILE`` to a Noto Sans JP font file to also check Japanese
PDF text. Portable tests explicitly use Matplotlib's bundled STIXGeneral font;
they do not rely on a silent replacement for the production font selection.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from plot_phase_cdf import Backend, open_record, plot_limits, reconstruct
from plot_phase_cdf_paper import create_figure, font_context, main
from test_plot_phase_cdf import fixture


def _pdf_reader(path: Path):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise unittest.SkipTest("pypdf unavailable; PDF structure/text was not checked")
    return PdfReader(path)


def _font_objects(page):
    """Include fonts inside Form XObjects, not only the top-level page."""
    seen_resources = set()

    def walk(resources):
        resources = resources.get_object()
        identity = id(resources)
        if identity in seen_resources:
            return
        seen_resources.add(identity)
        if "/Font" in resources:
            for reference in resources["/Font"].get_object().values():
                yield reference.get_object()
        if "/XObject" in resources:
            for reference in resources["/XObject"].get_object().values():
                obj = reference.get_object()
                if obj.get("/Subtype") == "/Form" and "/Resources" in obj:
                    yield from walk(obj["/Resources"])

    yield from walk(page["/Resources"])


class PaperPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record_dir = self.root / "record"
        self.pdf = self.root / "figure.pdf"
        self.preview = self.root / "preview.png"
        self.report = self.root / "report.json"

    def data(self, **kwargs):
        with open_record(self.record_dir) as record:
            return reconstruct(record, backend=Backend("cpu"), **kwargs)

    def invoke(self, *extra: str, output: Path | None = None):
        stdout, stderr = io.StringIO(), io.StringIO()
        args = [str(self.record_dir), "--out", str(output or self.pdf),
                "--device", "cpu", "--language", "en", "--latin-font", "STIXGeneral",
                "--width-mm", "85", "--height-mm", "52", "--dpi", "120", *extra]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def hashes(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.record_dir.iterdir() if p.is_file()}

    @contextlib.contextmanager
    def figure(self, data, *, limits=None):
        import matplotlib.pyplot as plt

        with font_context(language="en", latin_font="STIXGeneral"):
            fig, summary = create_figure(
                data, log_limits=limits or plot_limits(data, None, None),
                language="en", width_mm=85, height_mm=52, font_size=8, dpi=120,
            )
            try:
                yield fig, summary
            finally:
                plt.close(fig)

    def assert_heatmap_equal(self, fig, data):
        ax = fig.axes[0]
        artists = [*ax.images, *ax.collections]
        self.assertEqual(len(artists), 1, "The data panel must have one heatmap artist")
        rendered = np.ma.asarray(artists[0].get_array()).reshape(data.log10_pdf.shape)
        np.testing.assert_array_equal(np.ma.getdata(rendered), data.log10_pdf)
        np.testing.assert_array_equal(np.ma.getmaskarray(rendered), np.isneginf(data.log10_pdf))
        return artists[0]

    def assert_embedded_truetype(self, reader):
        fonts = [font for page in reader.pages for font in _font_objects(page)]
        self.assertTrue(fonts, "The PDF must contain fonts, not outlined text only")
        for font in fonts:
            self.assertNotEqual(font.get("/Subtype"), "/Type3")
            self.assertEqual(font.get("/Subtype"), "/Type0")
            self.assertIn("/ToUnicode", font, "PDF text requires a Unicode mapping")
            descendants = font["/DescendantFonts"]
            self.assertTrue(descendants)
            for reference in descendants:
                descendant = reference.get_object()
                self.assertEqual(descendant.get("/Subtype"), "/CIDFontType2")
                descriptor = descendant["/FontDescriptor"].get_object()
                self.assertIn("/FontFile2", descriptor)
                self.assertGreater(len(descriptor["/FontFile2"].get_object().get_data()), 0)

    def test_pdf_physical_size_embedded_fonts_and_greek_text(self):
        import matplotlib

        fixture(self.record_dir, nt=9, np_=11)
        # A user's normal plotting style must not silently crop the paper size.
        with matplotlib.rc_context({"savefig.bbox": "tight"}):
            code, stdout, stderr = self.invoke("--report", str(self.report),
                                               "--preview", str(self.preview))
        self.assertEqual(code, 0, stdout + stderr)
        self.assertTrue(self.preview.is_file())
        self.assertTrue(self.report.is_file())
        reader = _pdf_reader(self.pdf)
        self.assertEqual(len(reader.pages), 1)
        page = reader.pages[0]
        self.assertAlmostEqual(float(page.mediabox.width), 85 / 25.4 * 72, places=5)
        self.assertAlmostEqual(float(page.mediabox.height), 52 / 25.4 * 72, places=5)
        self.assert_embedded_truetype(reader)
        self.assertTrue(all("STIX" in str(font["/BaseFont"]) for font in _font_objects(page)),
                        "The requested Latin/math family must not be silently substituted")
        text = page.extract_text()
        self.assertIn("θ", text)
        self.assertTrue("φ" in text or "ϕ" in text, repr(text))
        self.assertIn("0", text)
        self.assertNotIn("synthetic_validation_only", text)
        self.assertNotIn("Saved-CDF reconstructed", text)
        self.assertNotIn("mass=", text)

    def test_nonuniform_angular_boundaries_and_crop_orientation(self):
        from matplotlib.collections import QuadMesh

        fixture(self.record_dir, nt=9, np_=11, kind="nonuniform")
        data = self.data(theta_range=(7.5, 133.0), phi_range=(-49.0, 73.0))
        with self.figure(data) as (fig, _):
            artist = self.assert_heatmap_equal(fig, data)
            self.assertIsInstance(artist, QuadMesh)
            coordinates = artist.get_coordinates()
            expected_phi, expected_theta = np.meshgrid(data.phi_edges_deg, data.theta_edges_deg)
            np.testing.assert_array_equal(coordinates[..., 0], expected_phi)
            np.testing.assert_array_equal(coordinates[..., 1], expected_theta)
            np.testing.assert_array_equal(fig.axes[0].get_xlim(), data.phi_range)
            np.testing.assert_array_equal(fig.axes[0].get_ylim(), data.theta_range[::-1])
            self.assertTrue(fig.axes[0].yaxis_inverted())

    def test_asymmetric_values_zero_mask_and_no_titles(self):
        fixture(self.record_dir, nt=9, np_=11)
        data = self.data()
        before = data.log10_pdf.copy()
        with self.figure(data) as (fig, _):
            self.assert_heatmap_equal(fig, data)
            self.assertFalse(fig.texts, "Paper figures must not contain debug footers")
            for ax in fig.axes:
                for location in ("left", "center", "right"):
                    self.assertEqual(ax.get_title(loc=location), "")
        np.testing.assert_array_equal(data.log10_pdf, before)

    def test_positive_underflow_survives_render_and_report(self):
        fixture(self.record_dir, nt=2, np_=2, kind="tiny")
        data = self.data()
        with self.figure(data, limits=(-405.0, 1.0)) as (fig, _):
            artist = self.assert_heatmap_equal(fig, data)
            self.assertTrue(math.isfinite(float(np.ma.getdata(artist.get_array()).flat[0])))
            self.assertEqual(artist.norm.vmin, -405.0)
            self.assertEqual(artist.norm.vmax, 1.0)
        before = self.hashes()
        code, stdout, stderr = self.invoke("--report", str(self.report),
                                           "--log-limits", "-405", "1")
        self.assertEqual(code, 0, stdout + stderr)
        summary = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(summary["positive_mass_product_underflows"], 1)
        self.assertEqual(summary["zero_probability_cells"], 0)
        self.assertFalse(summary["smoothed"])
        self.assertFalse(summary["renormalized"])
        self.assertFalse(summary["cdf_repaired"])
        self.assertAlmostEqual(summary["global_log10_pdf_min_positive"],
                               -400 - math.log10(np.pi), places=12)
        self.assertEqual(self.hashes(), before)

    def test_missing_latin_font_fails_without_fallback_or_outputs(self):
        fixture(self.record_dir, nt=3, np_=4)
        code, _, stderr = self.invoke("--latin-font", "__rainbow_no_such_font__",
                                     "--preview", str(self.preview), "--report", str(self.report))
        self.assertEqual(code, 1)
        self.assertIn("font", stderr.lower())
        for path in (self.pdf, self.preview, self.report):
            self.assertFalse(path.exists())

    def test_missing_japanese_font_file_fails_without_outputs(self):
        fixture(self.record_dir, nt=3, np_=4)
        code, _, stderr = self.invoke("--language", "ja", "--jp-font-file",
                                     str(self.root / "missing_noto_font.ttf"))
        self.assertEqual(code, 1)
        self.assertIn("font", stderr.lower())
        self.assertFalse(self.pdf.exists())

    def test_every_existing_destination_is_checked_before_any_write(self):
        fixture(self.record_dir, nt=3, np_=4)
        before = self.hashes()
        destinations = (self.pdf, self.preview, self.report)
        for occupied in destinations:
            with self.subTest(destination=occupied.name):
                occupied.write_bytes(b"retain this existing output")
                code, _, _ = self.invoke("--preview", str(self.preview), "--report", str(self.report))
                self.assertEqual(code, 1)
                self.assertEqual(occupied.read_bytes(), b"retain this existing output")
                for untouched in destinations:
                    if untouched != occupied:
                        self.assertFalse(untouched.exists())
                self.assertEqual(self.hashes(), before)
                occupied.unlink()

    def test_explicit_overwrite_replaces_all_destinations(self):
        fixture(self.record_dir, nt=3, np_=4)
        for path in (self.pdf, self.preview, self.report):
            path.write_bytes(b"previous output")
        code, stdout, stderr = self.invoke("--preview", str(self.preview), "--report", str(self.report),
                                           "--overwrite")
        self.assertEqual(code, 0, stdout + stderr)
        self.assertTrue(self.pdf.read_bytes().startswith(b"%PDF-"))
        self.assertTrue(self.preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertFalse(json.loads(self.report.read_text(encoding="utf-8"))["renormalized"])

    def test_record_directory_is_immutable_for_pdf_preview_and_report(self):
        fixture(self.record_dir, nt=3, np_=4)
        before = self.hashes()
        cases = [
            (self.record_dir / "figure.pdf", []),
            (self.pdf, ["--preview", str(self.record_dir / "preview.png")]),
            (self.pdf, ["--report", str(self.record_dir / "metadata.json")]),
        ]
        for output, extra in cases:
            with self.subTest(output=str(output), extra=extra):
                code, _, _ = self.invoke(*extra, "--overwrite", output=output)
                self.assertEqual(code, 1)
                self.assertEqual(self.hashes(), before)
                self.assertFalse(self.pdf.exists())

    def test_japanese_text_with_explicit_noto_font(self):
        configured = os.environ.get("RAINBOW_TEST_JP_FONT_FILE")
        if not configured:
            self.skipTest("Set RAINBOW_TEST_JP_FONT_FILE to test Japanese PDF glyphs")
        font_file = Path(configured)
        self.assertTrue(font_file.is_file(), "Configured Japanese test font does not exist")
        fixture(self.record_dir, nt=3, np_=4)
        code, stdout, stderr = self.invoke("--language", "ja", "--jp-font-file", str(font_file))
        self.assertEqual(code, 0, stdout + stderr)
        # Windows rejects deletion of open font streams. Linux permits unlinking
        # them, so also detect the retained descriptors where /proc is available.
        descriptors = Path("/proc/self/fd")
        if descriptors.is_dir():
            retained = []
            for descriptor in descriptors.iterdir():
                try:
                    target = os.readlink(descriptor)
                except OSError:
                    continue
                if "rainbow-paper-fonts-" in target and target.endswith(" (deleted)"):
                    retained.append(target)
            self.assertFalse(retained, "Temporary font streams remained open after export")
        reader = _pdf_reader(self.pdf)
        self.assert_embedded_truetype(reader)
        text = "\n".join(page.extract_text() for page in reader.pages)
        for label in ("方位角", "散乱角", "θ"):
            self.assertIn(label, text)
        self.assertTrue("φ" in text or "ϕ" in text, repr(text))


if __name__ == "__main__":
    unittest.main()
