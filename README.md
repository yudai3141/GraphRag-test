# GraphRAG-test

PTSD 対話エージェントの研究リポジトリ。

## 研究の目的

**認知科学的に妥当な「記憶の想起・更新メカニズム」を持つ対話エージェントを構築し、
PTSD とその回復過程（曝露療法による恐怖消去学習）を計算論的に表現・観察できるようにする**
ことを目指す。

一般的な RAG / GraphRAG は「クエリに近い文書を embedding で引く」検索器にとどまる。
本研究はここを一歩進め、

- **想起**を、人間の記憶に近いモデル（ACT-R のエピソード記憶＋拡散活性）で行い、
- **記憶そのものが対話を通じて変化していく**過程（＝学習・回復）まで扱える、

対話エージェントの土台をつくる。題材として、トラウマ記憶とその回復を扱う PTSD 当事者の
語りを用いる。

## アプローチと段階

土台となる GraphRAG から出発し、検索を認知モデルに置き換え、最終的に「記憶の更新（回復）」まで
扱えるよう、段階的に発展させる。

| 段階 | フォルダ | 位置づけ | 状態 |
|---|---|---|---|
| **Stage 0** | `simulation/` | GraphRAG 基準実装。テキスト→知識グラフ→ハイブリッド検索の土台 | 実装済み |
| **Stage 1** | `actr_initial_integration_v1/` | 検索を embedding 類似から **ACT-R activation**（時間減衰＋拡散活性）へ置換。記憶を「エピソード記憶／意味記憶」の2層で構造化 | 実装済み |
| **Stage 2** | （構想） | 記憶の**主観評価とその更新**。各記憶に「恐怖／安全」という競合する評価を持たせ、安全な対話（曝露）で安全側を強化し、恐怖の出力を弱める（恐怖消去学習） | 未着手・設計中 |

### Stage 2 の狙い（恐怖消去学習のモデル化）

曝露療法は「恐怖記憶を消す」治療ではなく、**「その記憶を思い出しても今は安全だ」という
新しい学習を上書き的に作る**治療（extinction learning）である。これをグラフ上では、

```
（恐怖のエッジは消さない）
   [トラウマ記憶] ── 恐怖(そのまま残す)
                 └─ 今は安全(対話のたびに強化 → 育つと出力が変わる)
```

と表現する構想。恐怖トレースは残したまま、安全な文脈での想起（曝露）を重ねて「今は安全」側の
活性を育て、応答が「恐怖」→「もう過去のこと」へ移り変わる過程を観察できるようにする。
appraisal をエピソード側／刺激（意味層）側のどちらに載せるか、更新を回数ベースにするか
予測誤差ベースにするか等は設計中。

---

## セットアップ

依存関係（uv プロジェクト）と `.env` は**リポジトリルートで共有**する。Neo4j の接続先も
全フォルダで共通。[uv](https://docs.astral.sh/uv/) で依存を管理している。

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

> 実行すると OpenAI / Neo4j に実接続し、API 利用料が発生する。各フォルダの `CHAT_MODEL` /
> `build_llm()` のモデル ID が実在するか確認してから実行すること。

---

## Stage 0 — `simulation/`（GraphRAG 基準実装）

テキストから知識グラフを自動構築して Neo4j に保存し、質問に対して
**構造化データ（グラフの近傍関係）** と **非構造化データ（ベクトル類似検索）** の
両方を文脈として組み合わせて回答を生成する、GraphRAG の基準実装。以降の段階がここから
どう変わるかを比較する土台になる。

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

```bash
# スクリプトとして実行
uv run python simulation/graphrag.py
```

```python
# 関数として利用
from simulation.graphrag import setup, ask

graph, chain = setup()                 # デフォルトで ptsd_story.txt を使用
answer = ask(chain, "What treatment helped the author recover from PTSD?")
print(answer)
```

主な関数: `setup(file_path)`（環境読み込み〜チェーン構築を一括）, `ask(chain, question)`,
`load_documents()`, `build_graph()`, `build_vector_index()`,
`export_graph_html()` / `show_graph()`（可視化）。

### グラフの可視化（任意）

```bash
uv sync --extra viz
```

`export_graph_html(output_path="graph.html")` でインタラクティブな HTML を書き出し、VSCode の
Simple Browser（コマンドパレットの `Simple Browser: Show`）またはブラウザで開く。Jupyter /
VSCode Notebook 上では `show_graph()` でウィジェット表示もできる。

---

## Stage 1 — `actr_initial_integration_v1/`（ACT-R 対話エージェント）

当事者本人（一人称ペルソナ）として、自分のエピソード記憶を **ACT-R activation**
（コンテキスト類似度 ＋ 時間経過に応じたベースレベル ＋ ノイズ）で想起しながら対話する
エージェント。Stage 0 との違いは検索の中核で、embedding 一発ではなく、

1. **ACT-R activation** で最も活性の高いエピソードを1つ選び（seed）、
2. その seed のエンティティを **意味層の連想（ASSOC）** でたどって関連エピソードを集める、

という二段想起で記憶を引く。レイヤード構成・活性化の式・実行手順・構築結果は
[`actr_initial_integration_v1/README.md`](actr_initial_integration_v1/README.md) を参照。

```bash
# 1. エピソード記憶グラフを構築（1回）
uv run python -m actr_initial_integration_v1.memory.builder
# 2. コンソールで対話
uv run python -m actr_initial_integration_v1.main
```
