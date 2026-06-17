# GraphRAG-test

PTSD 対話エージェントの研究リポジトリ。LangChain + Neo4j を使った
**GraphRAG**（Graph Retrieval-Augmented Generation）を土台に、本格的な対話
エージェントを構築していく。

## リポジトリ構成

依存関係（uv プロジェクト）と `.env` は**リポジトリルートで共有**する。
Neo4j の接続先も全フォルダで共通。

| パス | 内容 |
|---|---|
| `simulation/` | GraphRAG の基準実装（検証用）。`graphrag.py` ＋ 入力テキスト |
| `actr_initial_integration_v1/` | PTSD 対話エージェント（ACT-R 初期統合 v1）。ACT-R activation でエピソード記憶を検索する対話エージェント |
| `pyproject.toml` / `uv.lock` | 共有する依存関係の定義 |
| `.env` / `.env.example` | 共有する環境変数（OpenAI / Neo4j） |

## セットアップ

[uv](https://docs.astral.sh/uv/) で依存関係を管理している。

```bash
# 1. 依存関係をインストール
uv sync

# 2. 環境変数を設定（.env.example をコピーして値を埋める）
cp .env.example .env
```

`.env` に以下を設定する（`.env` は `.gitignore` 済みでコミットされない）。

| 変数 | 説明 |
|---|---|
| `OPENAI_API_KEY` | OpenAI の API キー |
| `NEO4J_URI` | Neo4j の接続 URI（例: `neo4j+s://xxxx.databases.neo4j.io`） |
| `NEO4J_USERNAME` | Neo4j のユーザー名 |
| `NEO4J_PASSWORD` | Neo4j のパスワード |
| `NEO4J_DATABASE` | 接続先のデータベース名（通常 `neo4j`）。インスタンスによってはインスタンス ID が既定 DB 名のことがある。不明なら `system` DB で `SHOW DATABASES` を実行して確認する |

---

## simulation/ — GraphRAG 基準実装

テキストから知識グラフを自動構築して Neo4j に保存し、質問に対して
**構造化データ（グラフの近傍関係）** と **非構造化データ（ベクトル類似検索）** の
両方を文脈として組み合わせて回答を生成する。

元は [Colab ノートブック](https://colab.research.google.com/github/nyanta012/demo/blob/main/GraphRAG.ipynb)
で、ローカル環境で扱えるよう各処理を関数化している。

参考: [Enhancing RAG-based applications accuracy by constructing and leveraging knowledge graphs](https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/)

### 仕組み

1. `ptsd_story.txt` を読み込み、トークン単位でチャンク分割する
2. `LLMGraphTransformer` で各チャンクからエンティティと関係を抽出し、知識グラフを構築して Neo4j に登録する
3. 質問が来たら以下を並行して取得し、文脈として結合する
   - **構造化検索**: 質問内のエンティティを全文検索で特定し、その近傍関係をグラフから取得
   - **非構造化検索**: ベクトルインデックスによるハイブリッド類似検索
4. 結合した文脈を LLM に渡して回答を生成する

### 構成

| ファイル | 内容 |
|---|---|
| `simulation/graphrag.py` | GraphRAG 本体（ロジック）。各処理は関数として呼び出せる |
| `simulation/ptsd_story.txt` | 知識グラフの元になる入力テキスト（PTSD 体験記）。出典: [PTSD UK - My own experience](https://www.ptsduk.org/why-ptsd-uk-is-here/my-own-experience/) |

### 使い方

#### スクリプトとして実行

```bash
uv run python simulation/graphrag.py
```

#### 関数として利用

```python
from simulation.graphrag import setup, ask

# 環境読み込み〜グラフ構築〜チェーン組み立てを一括で行う
graph, chain = setup()                 # デフォルトで ptsd_story.txt を使用
# graph, chain = setup("other.txt")    # 別のテキストを使う場合

# 質問する
answer = ask(chain, "What treatment helped the author recover from PTSD?")
print(answer)
```

#### 主な関数

| 関数 | 役割 |
|---|---|
| `setup(file_path)` | 環境読み込み〜チェーン構築までを一括実行し `(graph, chain)` を返す |
| `ask(chain, question)` | 組み立てたチェーンに質問して回答を得る |
| `load_documents(file_path)` | テキストを読み込みチャンク分割 |
| `build_graph(documents, llm)` | 知識グラフを構築し Neo4j に登録 |
| `build_vector_index()` | ハイブリッド検索用のベクトルインデックスを作成 |
| `export_graph_html(cypher, output_path)` | グラフをインタラクティブ HTML に書き出し（VSCode 向け） |
| `show_graph(cypher)` | グラフを可視化（Jupyter / VSCode Notebook 用） |

### グラフの可視化（任意）

可視化用の依存をインストールする。

```bash
uv sync --extra viz
```

**VSCode で見る（推奨）**: `export_graph_html()` でインタラクティブな HTML を書き出す。Notebook は不要で、通常のスクリプト実行で動く。ノードはラベル種類ごとに文字色を色分けし、右上に色の凡例を表示する。

```python
from simulation.graphrag import export_graph_html

export_graph_html(output_path="graph.html")
```

生成された `graph.html` を VSCode の Simple Browser（コマンドパレットの `Simple Browser: Show`）、またはブラウザで開くと、ノードをドラッグできるグラフが表示される。

**Notebook で見る**: Jupyter / VSCode Notebook 上では `show_graph()` でウィジェット表示もできる（`yfiles_jupyter_graphs` を使用）。

### 注意点

- `simulation/graphrag.py` の `build_llm()` で使用する OpenAI モデル名は、実在する ID か確認してから実行する。
- 実行すると OpenAI / Neo4j に実際に接続し、API 利用料が発生する。

---

## actr_initial_integration_v1/ — PTSD 対話エージェント

当事者本人（一人称ペルソナ）として、自分のエピソード記憶を **ACT-R activation**
（コンテキスト類似度 ＋ 時間経過に応じたベースレベル ＋ ノイズ）で想起しながら対話する
エージェント。レイヤード構成・式・実行手順は
[`actr_initial_integration_v1/README.md`](actr_initial_integration_v1/README.md) を参照。

```bash
# 1. エピソード記憶グラフを構築（1回）
uv run python -m actr_initial_integration_v1.memory.builder
# 2. コンソールで対話
uv run python -m actr_initial_integration_v1.main
```