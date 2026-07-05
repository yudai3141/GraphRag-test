# 恐怖構造モデル（Foa & Kozak）の設計メモ

v2 で意味層に導入する「恐怖構造（fear structure）」のノード／エッジ設計。
出発点の問題意識と要件は `tmp/v2_requirements.md`（git 管理外のローカルメモ）に基づく。
要点と元論文グラフは本ファイルに採録済み。

## 何を作るのか（v2 の芯）

要件の核は2つ：

- **ノードを増やすのではなく、エッジ構造（結びつき方）が主役**。
- **「信念」も「修正情報」も独立ノードにしない** → 信念＝刺激・反応・意味づけの
  張り方そのもの。

Foa & Kozak では恐怖記憶を「刺激要素・反応要素・意味要素」の3要素からなる
ネットワークとして捉える。PTSD らしさは、客観的な出来事ではなく、これらの
**過剰で自動的な結合**（一部の刺激だけで恐怖構造全体が発火する）に現れる。

## 元論文グラフからの観察

提示された論文グラフ（本ファイル末尾に採録）を読むと、`meaning` は一枚岩ではない：

| 実体 | 実際の役割 |
|---|---|
| `HEART_ATTACK`, `SUFFOCATION` | **具体的破局解釈**（その反応が何を意味するか。動悸→心臓発作） |
| `DANGEROUS`, `BAD` | **中核評価**（あらゆる解釈がここへ収束する終端） |
| `AFRAID` | 情動（実質は反応寄り → 本設計では Response に寄せる） |
| `INDEFINITELY` | 永続性の修飾（「ずっと続く」） |

エッジの流れは `反応 ⇄ 具体解釈 → 中核(DANGEROUS→BAD)` と、**具体から少数の中核へ収束**する。
さらに `動悸 ⇄ 心臓発作` のような双方向ループが自己増幅エンジン（動悸→心臓発作だと思う→さらに動悸）
になっている。

## ノードのタクソノミー（3ロール＋既存 Episode）

| ロール | 中身 | 例 |
|---|---|---|
| `Stimulus` 刺激 | 恐怖を喚起する対象・状況 | 上司, 怒鳴り声, 勉強, 試験, 電車 |
| `Response` 反応 | 身体・感情・行動の反応（情動も含む） | 動悸, 身体が固まる, 逃げたい, 恐怖 |
| `Meaning` 意味 | 本人にとっての意味づけ（下記2階層） | 攻撃される, 窒息して死ぬ / 危険, 無力 |
| `Episode`（既存・v1 と共通） | 過去に体験した出来事 | 「トンネルでパニック」 |

## Meaning は2階層（← v2 の設計の肝）

「危険」そのものが PTSD を表象するのではない。PTSD を表象するのは
**「少数の負の中核へ、あらゆる刺激・反応・具体解釈が密に収束し、良い記憶だけが
そこへ経路を持たない」構造**である。ゆえに Meaning にタクソノミーを入れる。

| 層 | 語彙 | 例 |
|---|---|---|
| **具体的意味づけ**（開・エピソード固有・LLM 自由抽出） | 制限なし | 窒息して死ぬ, 攻撃される, 見捨てられる, 自分のせい |
| **中核評価**（閉・少数・全体共有・固定シード） | 固定4種 | `危険 DANGER` / `悪い・自分はダメ BAD` / `無力・逃げられない POWERLESS` / `終わらない UNENDING` |

- 具体的意味づけ →（`ROLLS_UP`）→ 中核評価、へ集約する。
- 中核評価は Foa の `DANGEROUS`/`BAD` を核に、PTSD 認知（PTCI の
  「世界は危険／自分はダメ／制御不能」）を汲んで **4種に閉じる**。増減は後で可能。

### なぜこれで「PTSDらしい連想」が構造から出るか

負バイアスを手で入れなくても、構造から自然に出る：

- どの刺激・反応も最終的に少数の `危険/BAD` 中核へ流れ込む → 出力が何でも
  危険・無力の色に染まる。
- **良い記憶（ジム・回復など）は中核へのエッジを持たない** → 活性が届かず想起されない
  ＝「嫌な記憶は良い記憶とつながりにくい」がそのまま実現。

これが「一般人格では出ない発話」の源になる。

## エッジのタクソノミー（主役）

各エッジに `weight`（結合の強さ）を持たせる。恐怖構造の「病理」＝ weight が過剰な状態。
v2 Stage 2 での回復＝ weight を下げる／意味づけエッジを張り替える（**ノードは消さない**）。

### 主エッジ

| エッジ | 意味 | 例 |
|---|---|---|
| `Stimulus -[EVOKES]-> Response` | 刺激→反応 | 上司の指摘 → 身体が固まる |
| `Stimulus -[MEANS]-> Meaning(具体)` | 刺激→意味 | 上司の指摘 → 攻撃されている |
| `Response -[MEANS]-> Meaning(具体)` | 反応→意味 | 動悸 → 心臓発作で死ぬ |
| `Meaning(具体) -[ROLLS_UP]-> CoreValuation` | 具体→中核評価 | 心臓発作で死ぬ → 危険 |
| `Stimulus -[RECALLS]-> Episode` | 現在の刺激→過去の記憶 | 勉強 → 教育虐待の記憶 |
| `Episode -[LEADS_TO]-> Episode` | 嫌な記憶→嫌な記憶（負→負） | トンネル恐怖 → 事件の記憶 |

### 構造エッジ（論文にあり・自己増幅と文脈束縛のために任意で導入）

| エッジ | 意味 |
|---|---|
| `Stimulus -[CO_OCCURS]- Stimulus` | 状況の同時性（市場・人混み・帰り道） |
| `Response -[CO_OCCURS]- Response` | 反応の連鎖（動悸 ⇄ 過呼吸） |
| `Response -[MEANS]-> Meaning` の双方向化 | 動悸 ⇄ 心臓発作 の自己増幅ループ |

## 決定事項（2026-07-06 の議論で確定）

1. 中核評価の閉集合は **`危険 / BAD / 無力 / 終わらない` の4種**で開始（増減は後で可能）。
2. 具体 Meaning は **LLM に自由抽出させ、中核へ `ROLLS_UP` を貼らせる**（軽量版＝中核のみ、は採らない）。
3. データセットは **v1 と同じ `simulation/ptsd_story.txt`**（比較しやすさ優先）。
4. 情動（`AFRAID` 等）は Meaning ではなく Response に寄せる。

---

## 付録：元論文グラフ（参照用）

```json
{
  "nodes": [
    { "id": "SELF", "type": "stimulus" },
    { "id": "CROWD", "type": "stimulus" },
    { "id": "MARKET", "type": "stimulus" },
    { "id": "FAR_FROM_HOME", "type": "stimulus" },

    { "id": "WALKING", "type": "response" },
    { "id": "TACHYCARDIA", "type": "response" },
    { "id": "HYPERVENTILATION", "type": "response" },

    { "id": "HEART_ATTACK", "type": "meaning" },
    { "id": "SUFFOCATION", "type": "meaning" },
    { "id": "DANGEROUS", "type": "meaning" },
    { "id": "BAD", "type": "meaning" },
    { "id": "AFRAID", "type": "meaning" },
    { "id": "INDEFINITELY", "type": "meaning" }
  ],

  "before_emotional_processing": {
    "edges": [
      { "from": "SELF", "to": "WALKING" },
      { "from": "WALKING", "to": "CROWD" },
      { "from": "WALKING", "to": "MARKET" },
      { "from": "WALKING", "to": "FAR_FROM_HOME" },
      { "from": "MARKET", "to": "CROWD" },
      { "from": "MARKET", "to": "FAR_FROM_HOME" },
      { "from": "SELF", "to": "TACHYCARDIA" },
      { "from": "SELF", "to": "HYPERVENTILATION" },
      { "from": "TACHYCARDIA", "to": "HEART_ATTACK" },
      { "from": "HEART_ATTACK", "to": "TACHYCARDIA" },
      { "from": "HYPERVENTILATION", "to": "SUFFOCATION" },
      { "from": "SUFFOCATION", "to": "HYPERVENTILATION" },
      { "from": "TACHYCARDIA", "to": "HYPERVENTILATION" },
      { "from": "HYPERVENTILATION", "to": "TACHYCARDIA" },
      { "from": "TACHYCARDIA", "to": "AFRAID" },
      { "from": "AFRAID", "to": "TACHYCARDIA" },
      { "from": "HYPERVENTILATION", "to": "INDEFINITELY" },
      { "from": "INDEFINITELY", "to": "HYPERVENTILATION" },
      { "from": "HEART_ATTACK", "to": "DANGEROUS" },
      { "from": "SUFFOCATION", "to": "DANGEROUS" },
      { "from": "HEART_ATTACK", "to": "BAD" },
      { "from": "SUFFOCATION", "to": "BAD" },
      { "from": "DANGEROUS", "to": "BAD" },
      { "from": "BAD", "to": "DANGEROUS" },
      { "from": "AFRAID", "to": "BAD" },
      { "from": "INDEFINITELY", "to": "BAD" },
      { "from": "AFRAID", "to": "INDEFINITELY" },
      { "from": "INDEFINITELY", "to": "AFRAID" }
    ]
  },

  "after_emotional_processing": {
    "edges": [
      { "from": "SELF", "to": "WALKING" },
      { "from": "WALKING", "to": "CROWD" },
      { "from": "WALKING", "to": "MARKET" },
      { "from": "WALKING", "to": "FAR_FROM_HOME" },
      { "from": "MARKET", "to": "CROWD" },
      { "from": "MARKET", "to": "FAR_FROM_HOME" },
      { "from": "TACHYCARDIA", "to": "HYPERVENTILATION" },
      { "from": "HYPERVENTILATION", "to": "TACHYCARDIA" },
      { "from": "HEART_ATTACK", "to": "DANGEROUS" },
      { "from": "SUFFOCATION", "to": "DANGEROUS" },
      { "from": "DANGEROUS", "to": "BAD" }
    ]
  }
}
```

> **情動処理前後の差分**が v2 Stage 2（会話によるエッジ張り替え）の教師的イメージ：
> 治療後もノードは残り、`反応→具体解釈` や `SELF→反応` の病理的エッジが弱まり、
> 刺激・反応・意味づけが以前ほど自動的に結びつかなくなる。
