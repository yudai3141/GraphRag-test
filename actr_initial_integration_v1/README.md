# actr_initial_integration_v1

PTSD 対話エージェント（**ACT-R 初期統合 v1**）。

`simulation/` の GraphRAG が「embedding 類似度で検索」していたのに対し、本エージェントは
**ACT-R 風の activation 値**（コンテキスト類似度 ＋ 時間経過に応じたベースレベル ＋ ノイズ）で
エピソード記憶を検索する。グラフは「エピソード記憶」に対応する構造を持つ。

語り手は **当事者本人（一人称ペルソナ）**。ユーザーに話しかけられると、自分のエピソード記憶を
想起しながら一人称で語る。

## レイヤード・アーキテクチャ

可読性優先で、責務ごとに層を分離している（DI なし）。

```
domain/         ① ドメイン層      Episode / MemEntity / RetrievalCandidate（純粋なデータ）
infrastructure/ ② インフラ層      Neo4j アクセス・埋め込み・LLM（外部 I/O）
memory/         ③ 記憶構築層      テキスト → 構造化エピソードグラフ
retrieval/      ④ 検索層          二段想起（base.py=IF, activation.py=seed採点, recall.py=連想展開）
prompts/        ⑤ プロンプト層    一人称想起プロンプト
agent/          ⑥ アプリ層        対話オーケストレーション ＋ DEBUG 表示
config.py                          全パラメータの集約点
main.py                            コンソール対話ループ（エントリ）
```

依存の向きは上から下へ一方向（agent → retrieval/prompts → infrastructure → domain）。

## 記憶の2層構造（エピソード記憶 ＋ 意味記憶）

認知科学の **エピソード記憶 / 意味記憶（Tulving）** の区分に倣い、2層に分けて持つ。
GraphRAG のような自由なエンティティ抽出ではなく、**1 エピソード = 1 出来事**として構造化する
（"PTSD" のような概念そのものはエンティティにしない）。

```
エピソード層（episodic）
  (Episode { id, context(C-rep), event, sensory(S-rep), b_m, t_created, presentations, embedding })
     -[NEXT]-> (Episode …)                         # 物語順の時系列

束縛（binding: 1 エピソード→1つ以上のエンティティ）
  (Episode) -[INVOLVES|AT|FELT|...]-> (MemEntity)

意味層（semantic network）
  (MemEntity {name, type}) -[ASSOC {weight}]- (MemEntity)   # 共起ベースの連想ネット
```

- 埋め込みは Episode が持つ。
- `ASSOC` は「同一エピソードに共起したエンティティ同士」を結ぶ（重み=共起回数）。記憶構築時に生成。
- ラベルは `Episode` / `MemEntity`。simulation の `Document` / `__Entity__` とは名前空間を分けており、
  同じ Neo4j インスタンス上で共存する。

## 二段想起（検索の中核）

embedding 1 本で全エピソードを並べるのではなく、人間の想起に倣って 2 段で引く。

```
① seed 特定（エピソード記憶）  : ACT-R activation で最も活性の高いエピソードを1つ選ぶ … activation.py
② 連想展開（意味記憶）         : seed のエンティティ→ASSOC 連想先→それらを共有する別エピソード … recall.py
→ seed ＋ 連想エピソードを文脈に応答
```

これは ACT-R の spreading activation（手がかり中のエンティティ=source から、それを含む
チャンク=エピソードへ活性が流れる。`S_ji`・fan effect）を、エピソード層と意味層に分けて
素直に表現したもの。

### ① ACT-R activation（seed の採点）

`tmp-actr/run_final.py` の式に準拠（`retrieval/activation.py`）:

```
A(m) = B(m) + 類似度 × spreading重み + 瞬時ノイズ

  B(m)  : ベースレベル学習（時間減衰）  B = ln( Σ_k (t_now - t_k)^(-d) )
          さらに初期バイアス b_m を log-sum-exp で合成
  類似度: クエリ埋め込みとエピソード埋め込みのコサイン類似度
  ノイズ: ロジスティック分布の瞬時ノイズ
```

- `retrieval_threshold` 未満は想起対象外。A(m) 降順で上位 `TOP_K` を文脈にする。
- **B(m)** は各エピソードが持つ。初期はランダム割当（`config.BM_INIT_*`）。
  将来はグラフ作成時に時刻情報を抽出して `t_created` に反映する予定。
- 想起のたびに対象エピソードの `presentations` に現在時刻を追記 → 次回以降 B(m) が上がる（強化）。
- パラメータは `config.py` で一括変更可（`DECAY / RETRIEVAL_THRESHOLD / SPREADING_WEIGHT / INSTANT_NOISE`）。

> **pyactr について**: 将来の本格 ACT-R 拡張（プロダクションルール等）に備えて依存には入れているが、
> v1 のコア検索では使っていない。理由は、要件の spreading が embedding 類似度であり pyactr の
> シンボリック連想強度とは別物のため。`retrieval/base.py` の `ActivationEngine` を実装すれば、
> 後から pyactr バックエンド（`pyactr_engine.py`）へ差し替えられる。

## データフロー（1 ターン）

```
ユーザー発話
  → MemoryRecaller.recall()
      ① ActivationEngine.rank()      全エピソードの A(m) を計算 → seed を1つ特定
      ② store.associative_episodes()  seed のエンティティ連想で関連エピソード取得
  → debug_view.render()    ①活性ランキング→seed、②連想→関連エピソード を人間可読に出力
  → seed＋関連を一人称プロンプトへ → LLM 応答
  → 想起したエピソードを reinforce()（presentations 追記＝強化）
```

## 使い方

依存・`.env`・Neo4j 接続はリポジトリルートで共有（→ ルート README 参照）。

```bash
# 1. エピソード記憶グラフを構築（1 回。LLM 抽出 ＋ 埋め込み ＋ Neo4j 書き込み）
uv run python -m actr_initial_integration_v1.memory.builder

# 2. コンソールで対話
uv run python -m actr_initial_integration_v1.main
```

実行すると OpenAI / Neo4j に実接続し、API 利用料が発生する。`config.CHAT_MODEL` は実在モデル ID か
確認してから実行すること。

## 構築結果（ptsd_story.txt から）

| 要素 | 件数 |
|---|---|
| Episode（エピソード記憶） | 132 |
| MemEntity（エンティティ） | 331 |
| NEXT（時系列リンク） | 131 |
| ASSOC（共起連想・意味層） | 1596 |

### エピソード ↔ エンティティのマッピング例

```
ep_001 「PTSDと診断された。」
   → CAUSED:性的暴行(Object) | DURING:2015(Time) | FELT:恐怖(Emotion) | WITH:診断(Object)
ep_003 「電車での移動がすぐに問題になり、毎日の通勤が試練になった。」
   → WITH:電車(Object) | DURING:毎日(Time) | ABOUT:通勤(Concept)
ep_004 「閉じ込められたように感じて、パニック発作が起き始めた。」
   → AT:トンネル(Place) | WITH:電車(Object) | CAUSED:パニック発作(Symptom) | FELT:閉じ込められた感じ(Emotion)
ep_056 「毎日ジムへ行きアクアロビクスとズンバに参加した。」
   → AT:地元のジム(Place) | DURING:毎日(Time) | WITH:ズンバ(Treatment) | FELT:勇気(Emotion)
```

二段想起の例：「電車に乗るのは平気？」→ seed=ep_003（通勤）→ 共有エンティティ（`電車`/`毎日`）の連想で
ep_004（トンネルのパニック）・ep_056（毎日通ったジム＝回復）等を取得し、回復の記憶まで織り込んで応答する。

## グラフの可視化

```bash
uv run --extra viz python -m actr_initial_integration_v1.viz
# → actr_initial_integration_v1/graph.html を生成。ブラウザ/VSCode Simple Browser で開く
```

Episode＝青、MemEntity＝赤で色分け表示。既定では電車クラスタ周辺の部分グラフを描く
（全体は密なため）。別の範囲を見たいときは `viz.export_html(cypher=...)` に Cypher を渡す。

### 想起フローの可視化（1 例）

`flow_viz.py` は、1 クエリを実際に流して **クエリ→seed想起→エンティティ抽出→ASSOC連想→関連エピソード→応答**
の流れを段階レイアウトの図（`flow.html`）に書き出す（記憶は変更しない読み取り専用）。

```bash
uv run --extra viz python -m actr_initial_integration_v1.flow_viz "電車に乗るのは平気ですか？"
# → actr_initial_integration_v1/flow.html
```

段（左→右）: ①クエリ → ②seedエピソード(ACT-R) → ③seedのエンティティ → ④ASSOC連想エンティティ
→ ⑤関連エピソード → ⑥応答。例: seedの `電車/毎日` から連想が広がり、`ジム/ズンバ`(回復) や
`トンネル/パニック発作` のエンティティ経由で関連エピソードが引き出される様子が一目で分かる。
