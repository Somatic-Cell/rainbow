# 保存 CDF から論文用の図を出力する

`tools/plot_phase_cdf_paper.py` は、既存の CDF レコードを読み、軸、目盛り、
日本語ラベル、カラーバーを含む図を Matplotlib で出力します。
タイトル、実験条件の見出し、パネル番号、キャプションは図内に入れません。
これらは原稿側で組版します。

## 1. インストール

Python 3.10 以降を使います。リポジトリのルートで実行してください。
既存の Python 環境を使う場合、以下の `python` はその環境の実行ファイルに読み替えます。

```bat
python -m pip install -r tools\requirements-paper.txt
```

Linux Libertine の通常体・斜体・太字・太字斜体は、公式の TTF 配布から
ライセンスとともに `tools/fonts/libertine/` に同梱しています。
OS へのフォントインストール、LaTeX、PowerPoint、有料ソフトは不要です。
日本語には PC にインストール済みの **Noto Sans JP** を使用します。

CUDA 版 PyTorch は任意です。`--device auto` は既存の CDF 可視化ツールと
同じ規則で計算バックエンドを選び、選択結果を表示します。
明示的な `--device cuda` は CUDA が利用できない場合にエラーになります。
このコマンドは保存 CDF の読み取り・復元・描画だけを行い、ソルバを起動しません。

## 2. 日本語の論文用 PDF

以下の `D:\rainbow_dataset\records\i0000\w0000` を、手元のレコードの
ディレクトリに置き換えてください。`metadata.json` と次の 3 ファイルが必要です。

```text
phi_cdf.npy
theta_given_phi_cdf.npy
u_edges.npy
```

Windows のコマンドプロンプトでは次のように実行できます。

```bat
python tools\plot_phase_cdf_paper.py D:\rainbow_dataset\records\i0000\w0000 ^
  --out outputs\paper\phase_ja.pdf ^
  --language ja ^
  --width-mm 170 --height-mm 95 --font-size 8 ^
  --preview outputs\paper\phase_ja.png
```

PowerShell では行継続文字が異なるので、同じコマンドを 1 行にして実行できます。

```powershell
python tools/plot_phase_cdf_paper.py D:/rainbow_dataset/records/i0000/w0000 --out outputs/paper/phase_ja.pdf --language ja --width-mm 170 --height-mm 95 --font-size 8 --preview outputs/paper/phase_ja.png
```

出力は次の 3 点です。

- `phase_ja.pdf`: 原稿に挿入する図。文字と軸はベクタ、密なカラーマップは画像。
- `phase_ja.png`: 外観確認用。`--preview` を指定した場合に出力。
- `phase_ja.json`: 入力条件、検証結果、色の範囲、フォントなどの来歴。

JSON は `--report` で保存先を指定できます。
既存ファイルを置き換える場合は `--overwrite` を付けます。
レコード内への出力は拒否します。

### Noto Sans JP が検出されない場合

インストールの仕方によっては、Matplotlib がフォントを発見できないことがあります。
フォントファイルを直接指定できます。次のパスは例であり、実際の場所に置き換えます。

```bat
python tools\plot_phase_cdf_paper.py D:\rainbow_dataset\records\i0000\w0000 ^
  --out outputs\paper\phase_ja.pdf ^
  --jp-font-file "C:\Users\YOUR_NAME\AppData\Local\Microsoft\Windows\Fonts\NotoSansJP-Regular.ttf"
```

`NotoSansJP[wght].ttf` のような可変フォントも指定できます。
その場合は fontTools で `wght=400` の静的インスタンスを一時生成し、
Regular 相当の太さを固定します。元のインストール済みファイルは書き換えません。
不足したフォントや字形を DejaVu などへ黙って置き換えません。

Noto Sans JP は [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+JP)
からも取得できます。異なる日本語フォントを意図して使う場合は `--jp-font` に
その family 名を指定し、必要に応じて `--jp-font-file` も指定します。

## 3. 掲載寸法・色の範囲・ラベル

`--width-mm` と `--height-mm` は余白を含めた図全体の寸法です。
指定寸法を保つため、保存時の `bbox_inches='tight'` は使用しません。
本文のカラム幅に合わせて生成し、原稿中で大きく縮小しない使い方を勧めます。
例えば 85 mm 幅に掲載する場合は、最初から次の寸法で生成します。

```bat
python tools\plot_phase_cdf_paper.py D:\rainbow_dataset\records\i0000\w0000 ^
  --out outputs\paper\phase_column.pdf ^
  --width-mm 85 --height-mm 60 --font-size 8
```

カラーバーの既定表記は `p(omega) [sr^-1]` の数式組版です。
実際の図では omega はギリシャ文字、-1 は上付き、単位 sr は立体となります。
目盛りは `10` の累乗で表します。
日本語は通常の文字列、数式は `$...$` で指定でき、以下のように変更できます。
日本語と数式を混在させるカスタムラベルでは、Libertine で表示したい英数字や
単位も `$...$` の中に置きます。単位などの立体には `\mathrm{...}` を使います。

```bat
python tools\plot_phase_cdf_paper.py D:\rainbow_dataset\records\i0000\w0000 ^
  --out outputs\paper\phase_custom.pdf ^
  --xlabel "方位角 $\phi\;\mathrm{[deg]}$" ^
  --ylabel "散乱極角 $\theta\;\mathrm{[deg]}$" ^
  --colorbar-label "$p(\omega)\,[\mathrm{sr}^{-1}]$"
```

上記の数式入り文字列を PowerShell で渡す場合は、`$` の展開を防ぐため
ダブルクォートではなくシングルクォートで囲みます。

比較図では全図に同じ色の範囲を指定してください。
例えば密度の表示範囲を `10^-6` から `10^1` とする指定は以下です。

```bat
python tools\plot_phase_cdf_paper.py D:\rainbow_dataset\records\i0000\w0000 ^
  --out outputs\paper\phase_fixed_scale.pdf --log-limits -6 1
```

範囲外の正の値は端の色で表示し、その存在はカラーバーの延長表示と JSON に記録します。
値自体に下限を加えたり、確率密度を変更したりはしません。
`--decades 7` は全体の最大値から 7 桁分を表示する指定で、
異なるレコードの最大値が異なる場合には共通の色範囲にはなりません。

領域の拡大表示は、例えば次の指定で行います。

```bat
python tools\plot_phase_cdf_paper.py D:\rainbow_dataset\records\i0000\w0000 ^
  --out outputs\paper\phase_crop.pdf ^
  --theta-range 120 150 --phi-range -30 30 --log-limits -6 1
```

表示領域を変えても確率密度は再正規化せず、既定の色範囲は全球の値から決めます。
カラーマップは `--cmap`、ラスタライズの解像度は `--dpi` で変更できます。
解像度を増やしても、入力 CDF より細かい計算結果が増えるわけではありません。

英語の図は `--language en` で生成します。
`--out` の拡張子として `.pdf`、`.svg`、`.png` を選べます。
投稿用はフォントを埋め込む PDF を推奨します。
SVG は文字を文字列として保持するため、閲覧・編集先にも同じフォントが必要です。

## 4. CDF と可視化の意味

このツールは `plot_phase_cdf.py` の `open_record`、`Backend`、`reconstruct`、
`plot_limits` を再利用します。既存の v1/v2 CDF と数値検証は共通です。

- 横軸は入射軸周りの方位角 phi、縦軸は入射進行方向から測る散乱極角 theta。
- theta=0 度を上、180 度を下に表示します。
- 保存配列は `[phi, theta_edge]`、描画配列は `[theta, phi]` です。
- 保存された `u_edges` から角度境界を復元します。非等間隔格子を等間隔に置換しません。
- セル質量を `(4*pi/Nphi)*diff(u_edges)` で割った、立体角あたりの密度を描きます。
- ログ密度を直接色へ写像します。ログを二重に取ったり、先に実数密度へ戻して
  アンダーフローさせたりしません。
- 真のゼロ確率セルだけを白で表示します。有限の非常に小さな正値は保持します。
- 表示領域を限定しても CDF 全体の単調性、正規化、HG メタデータなどを検査します。

図から除いた条件や検証値は JSON に残します。`source_quality` と
`storage_processing` も参照してください。保存 CDF を正しく描けることと、
光学ソルバが物理的・数値的に十分な精度に達していることは別の検証です。

## 5. 原稿への挿入と subcaption

図のタイトル・説明は原稿側へ置きます。日本語の予稿では、例えば次の形です。

```latex
% プリアンブル
\usepackage{graphicx}

% 本文
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/phase_ja.pdf}
  \caption{出射方向に対する位相関数．色は立体角あたりの確率密度を表す．}
  \label{fig:phase}
\end{figure}
```

二つのパネルに小キャプションを付ける場合は、各 PDF を別々に出力して
`subcaption` が提供する `subfigure` 環境で組みます。
各環境内で `\includegraphics` の後に `\caption{...}` と `\label{...}` を置くと、
図の下に小キャプションが付き、(a)/(b) と参照番号を LaTeX が管理します。
ACM `acmart` の図では、図全体の `\Description{...}` も原稿に記入します。
日本語予稿の独自クラスでは、その投稿先のパッケージ・キャプション指定を優先します。

## 6. フォントの対応範囲と確認

`acmart` の本文は Libertine 系です。このツールも Linux Libertine で欧文・数字・
基本的なギリシャ文字を組み、日本語だけ Noto Sans JP に分担させます。
Matplotlib の MathText を使うため、`newtxmath` や `unicode-math` の
すべての字形選択・数式の間隔と厳密に一致するという意味ではありません。

PDF は TrueType フォントをサブセット埋め込みし、軸と文字を画像化しません。
`pdffonts` を使える環境では、出力に対して `pdffonts figure.pdf` を実行し、
埋め込みと Unicode 対応を確認できます。最終確認は実際の原稿への挿入後の
文字サイズ、上下左右の切れ、比較図どうしの色範囲も含めて行ってください。

回帰テストは GPU を使わずに実行できます。

```bat
python -m pip install pypdf
python -m unittest discover -s tests -p "test_plot_phase_cdf*.py"
```

日本語 PDF の追加テストは環境変数 `RAINBOW_TEST_JP_FONT_FILE` に
Noto Sans JP の実ファイルを指定して実行します。指定のない環境では
日本語のテストだけが明示的にスキップされます。

一次資料:

- [acmart 公式マニュアル](https://mirrors.ctan.org/macros/latex/contrib/acmart/acmguide.pdf)
- [Matplotlib: Fonts](https://matplotlib.org/stable/users/explain/text/fonts.html)
- [Matplotlib: MathText](https://matplotlib.org/stable/users/explain/text/mathtext.html)
- [fontTools: variable font instancer](https://fonttools.readthedocs.io/en/latest/varLib/instancer.html)
- [Linux Libertine 配布元](https://sourceforge.net/projects/linuxlibertine/files/linuxlibertine/5.3.0/)
