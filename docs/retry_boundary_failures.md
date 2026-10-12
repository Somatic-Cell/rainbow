# 共有境界の修正後に失敗条件だけを再計算する

対象は `datasets/water_a1_i300_380_830` である．添付された
`failures(3).json` では，27,300 条件中 26,923 条件が成功し，**377 条件**が
`cdf_input_boundary_pending` として残っている．未試行は 0 条件である．

`retry_boundary_failures.bat` は，既存のバッチランナーの世代管理と
`--retry-failed` を使用する．成功済みレコードを保持し，未解決の失敗条件を
各 1 回だけ再試行する．ソルバの許容誤差を緩めたり，不正な CDF を受理したりする
機能は追加していない．

以下のコマンドは Windows のコマンドプロンプト（`cmd.exe`）用である．
リポジトリのルートをカレントディレクトリとして実行する．

## 科学的条件

次の条件は既存の `gen.bat` と `configs/dataset_water_a1.json` に合わせている．

| 項目 | 値 |
| --- | --- |
| 半径 | 1.0 mm |
| 形状・物質 | 非球形雨滴，water |
| 温度・圧力 | 20 °C，101325 Pa |
| 入射角 | −90°～90°の 300 セルの中心，−89.7°～89.7° |
| 波長 | 380～830 nm，5 nm 刻み，両端を含む 91 波長 |
| 入射グリッド | 3001 × 3001 |
| 問い合わせグリッド | 1800 × 3600 |
| 保存 CDF グリッド | 900 × 1800 |
| 光学段階 | `diffraction` |
| 焦点位相オフセット | `0,0,0,0` |
| CDF 許容誤差 | L1 mass error: `1e-10`，lost probability mass: `1e-12` |
| 低解像度判定の扱い | 既存の実行と同じ，明示的な `--allow-underresolved` |
| 粗視化 TV の上限 | 既存設定と同じ `1.0` |

`--allow-underresolved` と TV 上限 `1.0` は元の計算の設定を維持するためのものであり，
角度解像度の収束や光学的な精度を保証する設定ではない．今回の再試行のために変更した
ものではない．設定を編集して再開しようとすると，既存の `batch_plan.json` との
不一致によりランナーが拒否する．

## 1. 再ビルドの前に現在のバイナリを保存する

データセット生成とビルドのプロセスを停止した状態で，リポジトリのルートから実行する．

```bat
retry_boundary_failures.bat backup
```

次の新しいディレクトリに，現在の EXE と必要な 5 個の GPU モジュールをコピーする．
コピー後は Windows の `fc /b` で，各コピーが元ファイルとバイト単位で一致することを
確認する．元の `batch_plan.json` と，存在する場合は `generation_history.json` も保存する．

```text
outputs\before_bilinear_boundary_fix\rainbow_generate.exe
outputs\before_bilinear_boundary_fix\modules\raindrop_trace.optixir
outputs\before_bilinear_boundary_fix\modules\patch_build.fatbin
outputs\before_bilinear_boundary_fix\modules\patch_query.optixir
outputs\before_bilinear_boundary_fix\modules\patch_optics.fatbin
outputs\before_bilinear_boundary_fix\modules\phase_cdf.fatbin
```

保存先が既に存在する場合は上書きせずに終了する．コピー途中で失敗した場合も，
コピー済みのファイルを自動削除しない．成功メッセージを確認するまでは再ビルドを
行わず，失敗した保存先や元ファイルの状態を確認する．

**保存するのは，修正版を再ビルドする前の，現在データセットに使用している EXE とモジュールである．**
既存の `backup.bat` に指定されている `outputs/before_regular_fp64_fix` は以前の世代の
保存先なので，今回の保存先として流用しない．新しい世代の登録時には，保存された
EXE と 5 モジュールすべての SHA-256 が，直前の有効な世代と一致することを
既存ランナーが検証する．単にコピーが一致したことと，直前の世代との一致は別の確認である．

既に再ビルドして旧バイナリを失っている場合，新しいバイナリを旧世代として登録したり，
計画ファイルのハッシュを書き換えたりしてはいけない．以前の保存先から，直前の世代に
対応する EXE とモジュール一式を確保する必要がある．

## 2. 修正版をビルドし，利用可能な環境で検証する

バックアップが成功した後に，修正されたソースからビルドする．

```bat
build.bat
```

既存の `test.bat` は別の `build-verify` ディレクトリで回帰テストを実行する．
CUDA／OptiX を使用できる環境では，GPU を含む該当テストの結果も確認する．

**CPU の共有境界回帰テストや Python の再開処理テストに合格しても，実 GPU 上で
377 条件がすべて解消したことにはならない．** 実データの解消件数は，以降の
再計算で確認する．また，世代登録は旧版と新版の数値的同等性を証明しない．
代表的な成功条件について旧版と新版を比較する場合は，既存データを上書きせず，
別の出力ディレクトリで計算する．

## 3. 必要なら最初の失敗条件を 1 件だけ再試行する

```bat
retry_boundary_failures.bat retry 1
```

添付された失敗一覧のままなら，最初の対象は `i0000_w0081`，入射角 −89.7°，
波長 785 nm である．最初に既存の成功レコードを検査するため，すぐに GPU の
再計算が始まらないことがある．予定件数は次のように表示される．

```text
verified_existing=26923; queued=377; failed_existing=377; this_run=1
```

初回の `retry` は，`bilinear_boundary_ownership_v1` という理由で修正版のバイナリを
明示的に登録する．登録済みの同じバイナリで再び実行した場合，履歴の重複登録はしない．

この 1 件が成功しても，残り 376 件が未解決なので **終了コードは 2** になる．
終了コードだけでこの 1 件の成否を判断せず，`progress.json` の
`completed=26924`，`failed=376` と，該当レコードの生成を確認する．
1 件が再び拒否された場合は，元の `completed=26923`，`failed=377` のままで，
該当する失敗履歴とログに今回の試行が追加される．

なお，既存ランナーの `--dry-run` は計算条件とコマンドの表示用である．
実行バイナリのハッシュ，世代履歴，既存レコードや実際の再試行件数の検査は行わない．
したがって，世代移行やデータの整合性の確認の代わりにはならない．

## 4. 残りの失敗条件を再試行する

```bat
retry_boundary_failures.bat retry
```

省略時の試行上限は，この固定データセットの総条件数である 27,300 である．
したがって，全失敗条件が対象になる．対象を選ぶのは `--retry-failed` なので，
成功済み条件や未試行条件を新たに計算することはない．1 件だけの確認を済ませていれば，
その成功した 1 件も自動的に除外される．

既存の成功済み `records`，`receipts`，元の `batch_plan.json` は保持される．
旧版による成功レコードは旧版の世代に，新たに成功したレコードは新版の世代に
対応付けられる．世代履歴には数値的同等性が未証明であることが記録される．
既存データの保持は，全条件を新版で再計算したことや，すべての光学値の精度が検証済みで
あることを意味しない．通常の再開検査はメタデータ，NPY の形状・サイズ，receipt の
整合性を確認するもので，既存 NPY の全要素を再検査するものでもない．

再試行中の制御は次のとおりである．

- 認識済みの CDF 構築前の数値的拒否は，データとして受理せずに次の条件へ進む．
- 失敗数・連続失敗数の上限は各 400 であり，元の 377 件を各 1 回試行できる．
- 未知のプロセス異常や不正な出力は，件数上限に達していなくても停止する．
- 認識済みの最終ディレクトリ rename 失敗は，`--recover-commits` によって
  保存済み CDF を検査した上で回復する．数値計算の失敗を回復する機能ではない．
- `--retire-uncommitted` は指定していない．未確定の `.part` などが見つかった場合は
  自動移動・削除せず停止するので，その診断を確認する．

添付の状態で 377 件を保存する配列ペイロードは約 4.90 GB である．設定中の
20 GiB の空き容量予約とレコードごとの余裕を含め，ランナーの事前見積もりは
約 26.42 GB の空き容量を要求する．既存データを複製する容量ではない．

## 完了の判定

全条件が完了した場合は `Dataset complete.` と表示され，終了コードは 0 になる．
`progress.json` と `failures.json` は次の状態になる．

| ファイル・項目 | 完了時の値 |
| --- | --- |
| `progress.json`: `completed` | `27300` |
| `progress.json`: `total` | `27300` |
| `progress.json`: `failed` | `0` |
| `progress.json`: `unattempted` | `0` |
| `progress.json`: `complete` | `true` |
| `failures.json`: `count` | `0` |

再試行対象を処理し終えても失敗が残れば，終了コードは 2 になる．これまでの
条件別失敗履歴とログは残るので，更新された `failures.json` とそこから参照される
`failures/i...._w....json`，`logs/i...._w....log` を使って次の原因調査を行う．
本バッチは `call` で既存バッチを呼び出し，その終了コードを保存・表示して，
呼び出し元にも同じ値を返す．

## 失敗が残った場合：実 GPU の詳細レポートを取得する

`rainbow_generate` の `--patch-failure-report` を使うと，CDF の構築に使用した
同じ GPU 計算結果から，失敗方向と関連パッチの診断情報を取得できる．
CDF が拒否された場合はその失敗を維持したままレポートを保存する．
クエリの失敗など，光学評価や CDF 構築より前に停止した場合にはレポートは生成されない．

このオプションには `RAINBOW_ENABLE_DIAGNOSTICS=ON` が必要である．
次のように別のビルド先を使用すれば，データセットに登録した
`build/Release` のバイナリを変更せずに診断できる．普段のビルドで SDK の場所や
CUDA アーキテクチャなどを指定している場合は，同じ指定も追加する．

```bat
cmake -S . -B build-boundary-report -DBUILD_TESTING=OFF -DRAINBOW_ENABLE_DIAGNOSTICS=ON && ^
cmake --build build-boundary-report --config Release --target rainbow_generate
```

例として，元の一覧の最初の失敗条件 `i0000_w0081`（−89.7°，785 nm）を計算する．
以下は元の計算と同じ物理条件，解像度，CDF 許容誤差をすべて指定している．
出力先とレポートはデータセットの外に分ける．保存先が既に存在すると `mkdir` が
失敗し，後続の計算を実行しないので，再取得時は未使用のディレクトリ名に変更する．

```bat
mkdir outputs\boundary_report_i0000_w0081 && ^
build-boundary-report\Release\rainbow_generate.exe ^
    --out outputs\boundary_report_i0000_w0081\data ^
    --material water ^
    --radius-mm 1.0 ^
    --wavelength-nm 785 ^
    --inclination-deg -89.7 ^
    --temperature-c 20 ^
    --pressure-pa 101325 ^
    --grid 3001 ^
    --query-theta 1800 ^
    --query-phi 3600 ^
    --cdf-theta 900 ^
    --cdf-phi 1800 ^
    --stage diffraction ^
    --device 0 ^
    --max-coarsening-tv 1.0 ^
    --cdf-max-l1 1e-10 ^
    --cdf-max-lost-mass 1e-12 ^
    --focal-offsets 0,0,0,0 ^
    --allow-underresolved ^
    --patch-failure-report outputs\boundary_report_i0000_w0081\patch_failure_report.json ^
    > outputs\boundary_report_i0000_w0081\run.log 2>&1
echo Exit code: %ERRORLEVEL%
```

別の未解決条件を調べる場合は，更新された `failures.json` に記載された入射角と
波長を使用し，診断ディレクトリ名も変更する．レポートの親ディレクトリは事前に必要で，
レポート本体とその `.part` は新規ファイルでなければならない．上の配置は，
レポートを `--out` とその `.part` ディレクトリの外に置くという条件も満たす．

確認用に提供するファイルは，生成された `patch_failure_report.json` と `run.log` である．
単独の `rainbow_generate` は計算失敗時に終了コード 1 を返すので，レポートが存在しても
CDF の成功とは限らない．逆に，修正によって光学的な不正値がなくなった条件では，
失敗方向が 0 件のレポートが生成されることがある．この別出力の確認計算は，
元のデータセットの成功件数や失敗履歴を変更しない．
