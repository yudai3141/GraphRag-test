# actr_foa_kozak_v2

PTSD 対話エージェント（**Foa & Kozak 恐怖構造 v2**）。

`actr_initial_integration_v1`（ACT-R でエピソードを想起）を土台に、想起の「連想」部分を
**Foa & Kozak の恐怖構造モデル**へ置き換えたエージェント。一見中立な入力でも、内部の
恐怖構造をたどって過去の嫌な記憶や「危険／無力」といった意味づけへ連想が伸び、
**PTSD 患者らしい（一般人格では出ない）想起**が生じることを狙う。

設計の背景・意思決定は [`docs/design_fear_structure.md`](docs/design_fear_structure.md)、
段階と実装計画は [`docs/roadmap.md`](docs/roadmap.md) を参照。

> **本フォルダの範囲＝v2 Stage 1**：恐怖構造による「連想の再現」まで（記憶＝エッジの更新はしない）。
> 会話によるエッジの張り替え（情動処理）は次段（v2 Stage 2）。

## v1 との違い（発話としてどう変わるか）

| | v1 (actr_initial_integration_v1) | v2 (本フォルダ) |
|---|---|---|
| 想起の起点 | ACT-R activation で seed エピソードを1つ選ぶ | クエリに近い**刺激(Stimulus)** を活性の入口にする |
| 連想の仕組み | エンティティ共起（ASSOC）で関連エピソード | **恐怖構造の拡散活性**（刺激→反応→意味→中核評価、刺激→過去記憶、嫌な記憶→嫌な記憶） |
| 出力の色 | 想起した出来事を素直に語る | 中立な話題でも**危険/無力の意味づけ**に引きずられる／良い記憶は想起されにくい |

## 記憶の構造（恐怖構造グラフ）

`Meaning` を「具体的意味づけ（開語彙）」と「中核評価（閉・少数）」の2階層に分けるのが肝。
PTSD らしさは、あらゆる刺激・反応・具体解釈が**少数の負の中核へ収束**し、良い記憶だけが
そこへ経路を持たない、という**構造**として表現される。

```
刺激 Stimulus ──EVOKES──▶ 反応 Response
    │  │                     │
    │  └──MEANS──▶ 具体的意味 Meaning ◀──MEANS──┘
    │                     │
    │                  ROLLS_UP
    │                     ▼
    │            中核評価 Core（危険 / BAD / 無力 / 終わらない）  ← 少数・全体共有・固定
    │
    ├──RECALLS──▶ 過去エピソード Episode（嫌な記憶）
    │                     │
    └──CO_OCCURS──刺激     └──LEADS_TO──▶ 別の嫌な記憶（負→負のみ）
```

- **ノード**: `FkStimulus` / `FkResponse` / `FkMeaning` / `FkCore` / `FkEpisode`
  （すべて `Fk` 接頭辞で v1 の `Episode`/`MemEntity`・simulation の `Document` と名前空間を分離）。
- **中核評価**は固定4種 `DANGER / BAD / POWERLESS / UNENDING`（`config.CORE_VALUATIONS`）。
- **エッジ** `EVOKES / MEANS / ROLLS_UP / RECALLS / LEADS_TO / CO_OCCURS` に `weight`（結合の強さ）。
- 拡散活性の入口は刺激なので、**刺激とエピソードにだけ埋め込み**を持たせる（他は構造でたどる）。

## 想起＝拡散活性（`retrieval/spreading.py`）

```
① 入口   : クエリ埋め込みに近い刺激を上位 SOURCE_TOP_K 個選ぶ（クエリはノード化しない）
② 拡散   : 恐怖構造のエッジを SPREAD_HOPS ホップたどる
           ・1 ホップごとに HOP_DECAY で減衰
           ・出力エッジで活性を分割（fan effect）× 関係ごとの重み REL_WEIGHT
           ・ROLLS_UP/LEADS_TO/RECALLS を強めにして「中核収束」「負→負連想」を効かせる
③ 集計   : 活性化した刺激・反応・意味・中核評価・過去エピソードを降順に収集
```

良い記憶（回復エピソード等）は恐怖構造へのエッジを持たない＝活性が届かず想起されない。
これが「嫌な記憶は良い記憶とつながりにくい」を**構造で**実現する。

## レイヤード構成（v1 と同じ方針・DI なし・可読性優先）

```
domain/         ドメイン層     FearNode / Episode / Edge / FearGraph / RecallResult
infrastructure/ インフラ層     Neo4j アクセス・埋め込み・LLM
memory/         記憶構築層     テキスト → 恐怖構造グラフ（extractor / builder）
retrieval/      検索層         拡散活性（spreading.py）
prompts/        プロンプト層   恐怖構造に引きずられた一人称想起
agent/          アプリ層       対話オーケストレーション ＋ DEBUG 表示
config.py                      全パラメータ（ラベル・中核語彙・拡散パラメータ）
main.py                        コンソール対話ループ
```

## 使い方

依存・`.env`・Neo4j 接続はリポジトリルートで共有（→ ルート README 参照）。

```bash
# 1. 恐怖構造グラフを構築（1回。LLM 抽出 ＋ 埋め込み ＋ Neo4j 書き込み）
uv run python -m actr_foa_kozak_v2.memory.builder

# 2. コンソールで対話
uv run python -m actr_foa_kozak_v2.main
```

実行すると OpenAI / Neo4j に実接続し、API 利用料が発生する。`config.CHAT_MODEL` /
`EXTRACT_MODEL` は実在モデル ID か確認してから実行すること。

## 可視化

```bash
# 恐怖構造グラフ全体（刺激=橙/反応=青/意味=赤/中核=紫/エピソード=緑）
uv run --extra viz python -m actr_foa_kozak_v2.viz          # → graph.html

# 1 クエリの拡散活性フロー（クエリ→入口→拡散→中核収束→想起エピソード→応答）
uv run --extra viz python -m actr_foa_kozak_v2.flow_viz "勉強って最近どう？"   # → flow.html
```

## 検証（成功条件）

- 客観的に中立な入力（例「勉強どう？」）でも、内部で過去の嫌な記憶・負の中核評価へ連想が伸びる。
- 良い記憶（回復エピソード）が構造的に想起されない（中核への経路を持たないため）。
- 拡散活性の核ロジックはスタブ埋め込みで単体確認済み
  （中立クエリ→反応/意味→BAD 収束→負エピソード想起、良い記憶は非到達）。
