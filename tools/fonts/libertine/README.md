# Linux Libertine for publication figures

The four unmodified TrueType fonts in this directory are from Linux
Libertine 5.3.0, by Philipp H. Poll. They are distributed here under the
SIL Open Font License 1.1; see `OFL-1.1.txt` for the original copyright,
reserved font names, and license. `LICENCE.txt` is the upstream licensing note.

Upstream release:

- <https://sourceforge.net/projects/linuxlibertine/files/linuxlibertine/5.3.0/>
- Archive: `LinLibertineTTF_5.3.0_2012_07_02.tgz`
- <https://downloads.sourceforge.net/project/linuxlibertine/linuxlibertine/5.3.0/LinLibertineTTF_5.3.0_2012_07_02.tgz>

| File | Face | SHA-256 |
| --- | --- | --- |
| `LinLibertine_Rah.ttf` | Regular | `06e2b67aa9a899e71f78b91da21cd0bba0fb1c5791fd3b58752ea25a4b9b79d7` |
| `LinLibertine_RIah.ttf` | Italic | `8006a620ae7487b45c8279cc51b676ec0209af7151fea00d69df44142171e7b6` |
| `LinLibertine_RBah.ttf` | Bold | `b044e2233db4f1a792e8422827679f3bc3b573c164597e9d3b03a63f123b4cdf` |
| `LinLibertine_RBIah.ttf` | Bold Italic | `830a780c355f8b577c7d34d47ccc45e5a99f2117a9e1ee4a37ee5b71de8eb8da` |

`plot_phase_cdf_paper.py` registers these fonts locally for its own rendering.
An OS-wide installation or a TeX installation is not required. Noto Sans JP
is not bundled: use an installed copy or the exporter's `--jp-font-file`
option. This directory must accompany the exporter when copying the tool.

The ACM `acmart` template uses the Libertine family. This exporter uses
Linux Libertine for Latin text, digits, and basic Greek symbols, and
Matplotlib MathText for mathematical layout. It does not claim to duplicate
all glyph choices or spacing of `newtxmath` or `unicode-math`.

TrueType outlines are intentional. OpenType fonts with CFF outlines are
not interchangeable with these files in the exporter's PDF embedding path.
