## MEMO
### 命名規則
このプロジェクトは以下のような規則で命名することにする:

|対象|規則|例|
|:--|:--|:--|
|namespace|`lower_camel_case`|`rainbow`|
|class|`UpperCamelCase`|`CudaContext`|
|struct|`UpperCamelCase`|``|
|enum / enum class|`UpperCamelCase`|``|
|型 alias|`UpperCamelCase`|``|
|関数|`lower_snake_case`|``|
|メンバ関数|`lower_snake_case`|``|
|CUDA kernel|`lower_snake_case`|``|
|変数|`lower_snake_case`|``|
|関数の引数|`lower_snake_case`|``|
|public struct field|`lower_snake_case`|``|
|メンバ変数|`lower_snake_case_`|``|
|マクロ|`RAINBOW_UPPER_SNAKE_CASE`|``|
|ファイル名|`lower_snake_case`|``|
|ディレクトリ名|`lower_snake_case`|``|
|CMake target|`rainbow_lower_snake_case`|``|

## クラス，構造体
所有権や不変条件をもつ型は class, そうでなければ struct として実装する

## 位相関数データセット：モデルと座標契約

このプロジェクトは、固定した雨滴形状・粒径・入射進行方向・真空波長について、
無偏光入射に対する出射方向分布を計算し、正規化CDFとメタデータを保存する。
点群生成、PyTorchへの転送、NF/MLPの学習は別プロジェクトの責務である。

`complete=true` は生成・保存の完了を示す。連続波動解との一致や角度収束の認定ではない。
`quality`、`storage_processing`、`diffraction_policy` を併せて参照する。

### 物理モデルと断面積

本データセットの輸送モデルは、**非吸収・幾何学的衝突モデル**である。
入射方向に対する投影面積を `Aproj` とすると、

- `cross_sections.scattering_mm2 = Aproj`
- `cross_sections.extinction_mm2 = Aproj`
- `cross_sections.absorption_mm2 = 0`

と**モデルとして定義**する。これらは未正規化の出射強度を積分して得た
Lorenz–Mie/Maxwell散乱断面積ではない。前方回折による消散断面積の増大を
別に足した値でもない。`cross_sections.wave_scattering_cross_section_computed=false`
で、この違いを明記する。単一散乱アルベドは1。

幾何データは `geometry.projected_area_mm2` と `geometry.projected_area_m2` に保存する。
`mm²` から `m²` への変換係数は `1e-6`。粒子数密度を `rho [m^-3]` とするなら、
このモデルの散乱・消散係数は `rho * Aproj_m2 [m^-1]`。
CDFやサンプリングPDFに投影面積を掛け直して正規化する必要はない。

面積計算にはソルバへ渡した形状係数・半径・入射ベクトルを使用する。
入射レイの採用セル数、包囲球の半径、CDFの解像度から面積を推定しない。
球は解析式、非球形は凸性の区間検査後、法線の投影面積の表面積分を使う。
方位角積分は解析的に行い、残る極角をCPU倍精度の適応Gauss–Kronrod積分で評価する。
積分誤差は推定値であり、厳密な誤差保証ではない。非凸・検証不能な形状は拒否し、
凸包や球の面積に置換しない。波長には依存しないが、現行の外部プロセス単位の
バッチでは各レコードでこの小さいCPU計算を再実行する。

水の屈折率は IAPWS R9-97、密度は IAPWS-95 の液体側圧力式から評価する。
`--wavelength-nm` は必須。外部媒質は絶対屈折率1、吸収虚部は0とする。
`material` に状態と評価した実際の屈折率を保存する。

### 方向ベクトルと粒子座標系

入出射の両ベクトルは**実際の光の進行方向**。

- `ki`：粒子へ向かって進む入射光の向き。
- `ko`：粒子から離れて進む出射光の向き。
- 粒子から光源を見る向きは `-ki`。`ki` と混同しない。

粒子座標は右手系、上向き `+y`、重力・形状の極軸は `-y`。
形状はy軸の周りに軸対称だが、上下対称を仮定しない。
`radius_mm` は等体積球の半径を基準とする形状パラメータであり、直径ではない。
形状の有限次数近似は `shape_coefficients` に記録する。

CLIの `--inclination-deg alpha` は次を設定する（角度をラジアンにして計算）。

```text
ki = (cos(alpha), -sin(alpha), 0)
alpha = 0     : +x 水平方向
alpha = +90   : -y 真下へ進む
alpha = -90   : +y 真上へ進む
```

全入射半球の鏡像を上下同一視して削減しない。
軸対称性で入射方位角の次元を取り除いた全入射極角範囲は
`alpha in [-90,90]` に対応し、`[0,180]` ではない。
`beta = 90 degrees - alpha` は形状の `-y` 軸から測った入射極角。

### 出射方向の theta と phi

`theta` は地理的緯度ではなく**入射軸からの散乱極角**。

```text
theta in [0, pi]
cos(theta) = dot(ki, ko)
theta = 0  : 前方出射（ko = ki）
theta = pi : 後方出射（ko = -ki、光源側へ戻る）
```

すべての(theta,phi)が出射方向を表す。縦軸の途中で入射/出射が切り替わらない。
`phi in [-pi,pi)` は入射軸の周りの方位角で、周期境界の `-pi` と `+pi` は同じ方向。
極 `theta=0,pi` ではphiは未定義であり、任意の一つの代表値で扱う。

`e0` を横基底、`e1 = cross(ki,e0)` として、

```text
ko = cos(theta)*ki + sin(theta)*cos(phi)*e0 + sin(theta)*sin(phi)*e1
```

`phi=0` は `e0` 側、`phi=+90 degrees` は `e1` 側。
現在のcanonical入射平面xyでは `e0=+z`、`e1=(-sin(alpha),-cos(alpha),0)`。
phiの±90度はxy鏡映平面の二つの子午線であり、「入射方位角が±90度」という意味ではない。
この平面の鏡映は `phi -> wrap(pi-phi)`。`phi=+90` と `phi=-90` の値自体の一致は要求しない。

20度入射の検算（理想的な基底の説明値）：

| 粒子座標の出射方向 | (theta, phi) [degrees] |
|---|---|
| +z | (90, 0) |
| -z | (90, -180) |
| -y | (70, +90) |
| +y | (110, -90) |
| +x | (20, -90) |

### `sampling_frame_columns` の行列配置

名前に `columns` とあるが、JSONの外側の配列は**通常の行列の行**。
基底が列に入る行列 `M = [e0 e1 ki]` を保存している。

```python
M = np.asarray(metadata["sampling_frame_columns"], dtype=np.float64)
e0, e1, ki = M[:, 0], M[:, 1], M[:, 2]

# 単一の列ベクトル
ko_particle = M @ ko_local
ko_local = M.T @ ko_particle

# 各行が一方向の [B,3] 配列
ko_particle_batch = ko_local_batch @ M.T
ko_local_batch = ko_particle_batch @ M
```

JSONの各内部配列を列ベクトルだと思って `column_stack` しない。
ソルバのFP32基底をFP64で直交化した利用側の基底がM。
`query_frame_axis/e0` はソルバの元の値を記録する照合用。
読み込んだレコードの利用に、別規約のONBを生成して上書きしない。

### CDFのファイルと配列軸

一つのディレクトリが**一つの入射条件・波長**に対応する。
入射角や波長に対する同時確率分布ではない。

```text
metadata.json
a: phi_cdf.npy                 shape=(Nphi+1,)
T: theta_given_phi_cdf.npy     shape=(Nphi,Ntheta+1)
U: u_edges.npy                 shape=(Ntheta+1,)
```

三配列は標準NPY、little-endian FP64 (`<f8`)、C-order。
900×1800出射セルなら `T.shape == (1800,901)`。
`T[j,i]` のjはphiセル、iはtheta/u境界。flat offsetは
`j*(Ntheta+1)+i`。画像の画素配列 `[theta,phi]` とは軸順が逆。
二変数の同時CDFではなく、phi周辺CDFとtheta条件付きCDFの組である。

`U[i]=(1-cos(theta_edge[i]))/2` は非等間隔。`i/Ntheta` で置き換えない。
phiの境界は `-pi+2*pi*j/Nphi`。
CDFは境界値を保存し、ソルバのクエリはセル中心を評価する。
保存セルはクエリセルの立体角質量を集約したもので、保存セル中心における
真の連続的な位相関数を保存したものではない。

セル確率と立体角PDFは、次で復元する。

```python
q = np.diff(a)                             # [Nphi]
b = np.diff(T, axis=1)                     # [Nphi,Ntheta]
mass = q[:, None] * b
solid_angle = (4*np.pi/Nphi) * np.diff(U)  # [Ntheta]
pdf_omega = mass / solid_angle[None, :]    # [Nphi,Ntheta], sr^-1
plot_pixels = pdf_omega.T                  # 描画するときだけ転置
```

この復元に追加の再正規化・平滑化・微小値のfloorは不要。
零確率のphi列の条件付きCDFは `U` に設定するが、周辺確率が0なので選択されない。
CDFの平坦区間を逆引きして0で割らないようにする。

### 学習側との確率測度の契約

学習座標を `[u,v]` とするなら、

```text
u = (1-cos(theta))/2
v = (phi+pi)/(2*pi)
dOmega = 4*pi du dv
pdf_uv = 4*pi * pdf_omega
```

NFの `[u,v]` と、CDFの保存軸 `[phi,theta]` は異なる。
セル内はuとvについて一様（立体角密度が一定）。thetaを線形補間して生成すると
異なる分布になる。vは周期的、uは非周期。球面上の二つの極ではvが縮退する。
コードを別プロジェクトに実装する場合も、この規約を維持する。

`hg.g` は**保存CDFの分布**に対する `E[dot(ki,ko)]`。
セル内の平均余弦 `1-U[i]-U[i+1]` を使った積分値であり、
`g_source`（縮約前）とは用途が違う。正のgはtheta=0への前方性。

### 論文用の位相関数プロット

保存 CDF から、目盛り・軸ラベル・カラーバーを含む論文用 PDF を出力できる。
日本語は Noto Sans JP、数字と基本的な数式は同梱の Linux Libertine を使用する。
図内のタイトルやパネル番号は付けず、原稿側でキャプションを組版する。

```bat
python -m pip install -r tools\requirements-paper.txt
python tools\plot_phase_cdf_paper.py D:\rainbow_dataset\records\i0000\w0000 --out outputs\paper\phase.pdf --language ja --width-mm 170 --height-mm 95 --font-size 8
```

PDF と同じ stem の JSON に入力条件、正規化の検証結果、色の範囲、フォントを保存する。
既存の CDF 読み込み・復元を再利用し、確率密度の平滑化や再正規化は行わない。
掲載寸法、共通カラースケール、Noto Sans JP の明示指定、LaTeX への挿入方法は
[論文用図の手順](tools/PAPER_FIGURES.md) を参照。

### レンダラへの接続

粒子から光源を見る方向を `sL`、粒子からカメラを見る方向を `sC` とすると、
問い合わせる組は `p(ko=sC | ki=-sL)`。両方向を点から離れる向きとして
扱うAPIに、符号を変えずこのCDFを渡さない。

粒子からworldへの剛体回転をRとすれば、`ki_particle=R.T@ki_world`、
`ko_particle=R.T@ko_world`。固体角は回転で保存される。
入射方位角を軸対称性でcanonical xy平面へ移す場合は、**入出射を同じy軸回転で**移す。
入射方向だけを回転して出射方向をそのままにしない。

本データの正規化は入射固定・出射に関するもの。カメラ側からの逆追跡で
固定する引数を交換する場合、単なる `p(ki|ko)=p(ko|ki)` は仮定しない。
輸送応答は `K(ki,ko)=Aproj(ki)*p(ko|ki)`。
相反性の検査は `K(ki,ko)` と `K(-ko,-ki)` を比較する。
面積データを追加したことだけで近似ソルバの相反性が証明されたわけではない。

### 一括生成

`configs/dataset_water_a1.json` で粒径・波長範囲・間隔・入射範囲を指定する。
`generate_dataset.bat` はPython標準ライブラリだけを使う起動口。
デフォルトはdry-runであり、確認後に明示的な `--execute` で開始する。

```bat
rem 計画と容量だけ確認（ファイル作成なし）
generate_dataset.bat --config configs\dataset_water_a1.json --dry-run

rem まず1レコードだけ。未解像の回折警告を承知する場合だけ最後のフラグを使う。
generate_dataset.bat --config configs\dataset_water_a1.json --execute --limit 1 --allow-underresolved

rem 同じ計画・同じバイナリで残りを生成
generate_dataset.bat --config configs\dataset_water_a1.json --execute --resume --allow-underresolved
```

既定の角度条件は `[-90,90]` を900区間に分けた中心
`-89.9, -89.7, ..., +89.9 degrees`。厳密な両極は含まれない。
波長は `380..830 nm`、5 nm間隔、両端を含め91波長。
900×91=81,900レコード、FP64三配列の総量は約1.0644 TB（ヘッダ・ログ等を除く）。

波長範囲だけ変更する場合：

```bat
generate_dataset.bat --config configs\dataset_water_a1.json --wavelength-range 380 720 5 --out-root D:\rainbow_380_720 --dry-run
```

終点に間隔が割り切れない設定は拒否する。FP32で同一になってしまう波長も拒否する。
既存データの設定・バイナリが違う場合は別の出力ルートを使用する。
`--limit` は今回処理する未生成レコード数だけを制限し、データセットの定義は変えない。

```text
output_root/
    batch_plan.json             # 条件軸・設定・実行ファイル/module SHA256
    progress.json
    records/i0000/w0000/        # incident_indexが外側、wavelength_indexが内側
    receipts/i0000_w0000.json   # 正常終了時のmetadata hash・NPYサイズ
    logs/i0000_w0000.log
```

再開時には条件と実行ファイル/moduleのハッシュ、metadata、NPYのヘッダとサイズを照合する。
NPY全内容のbit-rot検出やCDFの全量再検査はバッチの再開処理では行わない。
必要なデータ検査は別途 `PhaseRecord(validate=True)` などで実行する。
GPU/ドライバが変わった場合のビット一致まで保証するものではない。
失敗時は最初の失敗で停止し、未処理条件を正常扱いして飛ばさない。
`.part` やレシートのない既存レコードを勝手に削除・採用しない。
クラッシュ後の `.batch.lock` は実行中プロセスがないことを確認してから手動で削除する。

**現在のバッチは一条件一プロセスの逐次実行**。GPUコンテキストや光学バッファを
プロセス間で再利用する高速化、問い合わせバッチ化によるVRAM削減は実装していない。
メモリ不足への対応として解像度を自動で下げる処理もない。
付属configのクエリ1800×3600は既知の比較条件であり、収束済みGTの認定設定ではない。
`allow_underresolved=false` が既定。許可するなら上の例のように明示する。
`maximum_coarsening_tv=1` は構造的上限であり、研究の精度基準ではない。

### 研究での位置付け

Sadeghi et al. (2012)に基づく近似ソルバとして、有限の経路族、パッチ補間、
非正則枝の有限面積近似、局所回折フィルタを含む。
追加の全球ガウシアンは通常生成では行わない。
NFの参照分布として使用するときも、光学モデルの誤差・数値離散化誤差・
保存時の縮約誤差・NF近似誤差を分けて報告する。
球形比較、格子収束、入射角/波長の中間条件、代表的な相反性・対称性の確認は、
`complete=true` やビルドテスト成功とは別の研究上の検証である。
