# 対称翼型・前縁半径設計ツール

翼厚・最大厚み位置・前縁半径を指定して対称翼型を生成し、XFOIL解析、人体クリアランスによる必要翼弦算出、CAD出力につなぐPythonコードです。

## 最初に読むもの

- [前縁半径の定義・実装・統合ポイント](01_引継ぎ/他メンバー共有用_前縁半径と統合ポイント.txt)
- [技術討議の履歴](01_引継ぎ/翼型生成_技術討議.md)：実験結果、限界、メンバー間の見解の相違。記載された外部成果物はこのパッケージには含みません。

## フォルダ構成

```text
README.md                    導入・受け渡し方法
01_引継ぎ/                   統合説明と技術判断の経緯
v2_funcgen_cockpit/
  core/                      形状生成・DB・解析・選定
  tests/                     テストと研究用ベンチマークのコード
  *_cli.py                   操作入口
  library_24h.py             時間制限付きライブラリ生成
  requirements*.txt          依存ライブラリ
02_データ/
  設定/                      計算範囲のJSON
  人体プロファイル/          標準例と入力テンプレート
  v2/                        DB・生成翼型の配置先（別途受け渡し）
03_CAD出力/                  CADの生成先
05_研究記録/                 研究結果の生成先
```

`core/paths.py`がこの相対配置を使うため、上記のフォルダ名を保ってください。

## セットアップと確認（Windows PowerShell）

Python 3.14.3で開発・検証しています。

```powershell
cd v2_funcgen_cockpit
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe grid_expander_3d_cli.py --dry-run
```

形状生成だけを利用する場合：

```python
from core.geometry import generate_funcgen_airfoil
x, y = generate_funcgen_airfoil(24.0, 42.0, kle=1.2, method="cst")
```

この例は`v2_funcgen_cockpit`を作業ディレクトリまたはPythonのモジュール参照先にして実行します。学習済みモデルやAI APIは必要ありません。

## データと外部実行ファイル

- 空力解析には、利用条件を確認して入手したXFOILを`v2_funcgen_cockpit/xfoil.exe`へ配置してください。実行ファイルは同梱していません。
- 既存DBを用いた選定には、別途受け渡したCSVを`02_データ/v2/v2_Fairing_Airfoil_Database.csv`へ配置してください。DBを持たない状態では実データによる翼型選定はできません。
- 生成済み座標は必要に応じて`02_データ/v2/airfoils/`へ配置してください。キャッシュは再生成できます。
- 大量計算は自動実行されません。格子CLIは標準でプレビューです。範囲と計算時間を確認してから`--run`を指定してください。`library_24h.py`も明示起動が必要です。

## GitHubへのアップロード

**このフォルダの中身をリポジトリ直下にアップロード**してください。`.gitignore`も含めます。仮想環境、巨大DB、キャッシュ、計算ログ、過去CAD、旧版、エージェント連携用の作業状態は除外しています。

Gitを使用する場合、このフォルダで`git init`し、`git add .`、`git commit`の後、自分のリポジトリに接続してpushします。GitHubのWeb画面を使う場合も、このフォルダの内容を選択してください。

公開範囲に応じた利用許諾はメンバー間で決定してください。このパッケージには新たなライセンスを設定していません。

## 計算結果の扱い

Warp/CST、形状版、Re、迎角を区別してデータを統合してください。前縁半径倍率`kle=1.2`が普遍的な空力最適値であることを示すものではありません。研究用ベンチマークの単一候補の結果を、全形状やニューラル手法全般に一般化しないでください。
