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
| **Stage 2** | `actr_foa_kozak_v2/` | 想起の連想を **Foa & Kozak の恐怖構造モデル**へ置換。刺激・反応・意味づけの結合で、中立な入力でも過去の嫌な記憶や危険の意味づけへ連想が伸びる（PTSD らしさ）。将来は会話でエッジを張り替える（恐怖消去学習） | 実装中（Stage 1＝連想再現は実装済み） |
| **評価** | `caps5_graph_structure_eval_v3/` | 構築した恐怖構造グラフ自体の妥当性を定量評価（実験1）。この過程でオントロジーを**理論に忠実な形へ作り直し**（中核層の撤去・意味の開語彙化・二層化・無向対称の伝搬） | 実装済み |

### Stage 2 の狙い（恐怖構造と恐怖消去学習のモデル化）

PTSD らしさは客観的な出来事ではなく、**刺激・反応・意味づけの過剰な結合**（一部の刺激だけで
恐怖構造全体が発火する）に現れる（Foa & Kozak）。理論忠実化後のオントロジーは**二層**で表現する：

```
恐怖構造レイヤ（Foa/Lang）:
  刺激 ──EVOKES──▶ 反応 ──MEANS──▶ 意味（開語彙・少数カテゴリに丸めない）
  刺激 ──CO_OCCURS──▶ 別の刺激 ／ 刺激 ──SIMILAR──▶ 意味的に近い刺激（般化）
エピソード記憶レイヤ（二重表象理論）:
  刺激 ──RECALLS──▶ 過去エピソード ──BINDS──▶ その刺激-反応-意味（記憶＝S-R-Mの束）
```

結合は連想（無向・対称）とし、エッジ重みは一律とする（連想強度の差・方向の非対称はモデル化しない）。
怖がり度の指標は**反応要素の活性**から読む（生体情報理論）。曝露療法（恐怖消去学習）は、ノードを
消すのではなくこの**病理的なエッジを弱め、意味づけを張り替える**過程としてモデル化する（＝v2 Stage 2）。
理論忠実化の経緯・評価の詳細は [`caps5_graph_structure_eval_v3/`](caps5_graph_structure_eval_v3/README.md) と
[`caps5_graph_structure_eval_v3/docs/00_overview.html`](caps5_graph_structure_eval_v3/docs/00_overview.html) を参照。

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

---

## Stage 2 — `actr_foa_kozak_v2/`（Foa & Kozak 恐怖構造エージェント）

Stage 1 を土台に、想起の「連想」部分を **Foa & Kozak の恐怖構造モデル**へ置き換えた
エージェント。刺激・反応・意味づけの結合（`EVOKES/MEANS/ROLLS_UP/RECALLS/LEADS_TO`）を
たどる**拡散活性**で想起するため、一見中立な入力でも過去の嫌な記憶や少数の負の中核評価
（危険/BAD/無力/終わらない）へ連想が伸び、**PTSD 患者らしい想起**が生じる。良い記憶は恐怖構造へ
経路を持たず想起されにくい。設計・実行手順は
[`actr_foa_kozak_v2/README.md`](actr_foa_kozak_v2/README.md) を参照。

- **v2 Stage 1（本フォルダの現状）**: 恐怖構造による連想の再現まで（実装済み）。
- **v2 Stage 2（今後）**: 会話によるエッジの張り替え（恐怖消去学習）。

```bash
# 1. 恐怖構造グラフを構築（1回）
uv run python -m actr_foa_kozak_v2.memory.builder
# 2. コンソールで対話
uv run python -m actr_foa_kozak_v2.main
```

---

## 評価 — `caps5_graph_structure_eval_v3/`（恐怖構造グラフの定量評価＋理論忠実化）

`actr_foa_kozak_v2` の恐怖構造グラフ自体が「トラウマに関連するほど強く、無関係には弱く」
反応するか（＝グラフの妥当性）を、トラウマとの近さで層化した刺激バッテリー（100発話）で
定量評価する（論文の実験1）。この検討過程で、Foa & Kozak / Lang の一次理論に当たり直し、
オントロジーを理論に忠実な形へ作り直した（中核層の撤去・意味の開語彙化・二層化・無向対称）。
設計・結果・図解は [`caps5_graph_structure_eval_v3/README.md`](caps5_graph_structure_eval_v3/README.md) と
`caps5_graph_structure_eval_v3/docs/`（`00_overview.html` が入口）を参照。

論文（日英）は [`caps5_graph_structure_eval_v3/paper/`](caps5_graph_structure_eval_v3/paper/)。再現手順は
[`caps5_graph_structure_eval_v3/README.md`](caps5_graph_structure_eval_v3/README.md) 参照。骨子だけ示すと：

```bash
# グラフ構築 → 測定 → 集計/図 → 報告書
uv run python -m caps5_graph_structure_eval_v3.pipeline.build_snapshot \
    simulation/ptsd_story.txt caps5_graph_structure_eval_v3/data/graph_exp_ptsd.json
uv run python -m caps5_graph_structure_eval_v3.pipeline.runner \
    caps5_graph_structure_eval_v3/data/graph_exp_ptsd.json \
    caps5_graph_structure_eval_v3/data/cue_battery.json \
    caps5_graph_structure_eval_v3/results/metrics_ptsd.csv
uv run --with matplotlib python -m caps5_graph_structure_eval_v3.pipeline.analyze
```
