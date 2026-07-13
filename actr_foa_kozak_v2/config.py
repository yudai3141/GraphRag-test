"""エージェント全体の設定（v2: Foa & Kozak 恐怖構造）。

モデル名・Neo4j のラベル名前空間・中核評価の語彙・拡散活性のパラメータをここに集約する。
値を変えたいときは基本このファイルだけ見ればよい（可読性優先・DI なし）。
設計の背景は docs/design_fear_structure.md / docs/roadmap.md を参照。
"""

import os

from dotenv import load_dotenv, find_dotenv

# --- パス ---------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# 恐怖構造の元になるテキスト（v1 と同じものを使う＝比較しやすさ優先）
SOURCE_TEXT = os.path.join(_ROOT, "simulation", "ptsd_story.txt")


def load_env() -> None:
    """リポジトリルートの .env を読み込む（どの cwd からでも見つける）。"""
    load_dotenv(find_dotenv())


# --- モデル -------------------------------------------------------------
CHAT_MODEL = "gpt-5.4"            # 応答生成（v1 と同じ既定。実在 ID か要確認）
EXTRACT_MODEL = "gpt-5.4"         # 恐怖構造の抽出（構造化出力）
EMBED_MODEL = "text-embedding-3-small"

# --- Neo4j ラベル名前空間（v1: Episode/MemEntity, simulation: Document/__Entity__ と分離） ---
# すべて "Fk"(Foa & Kozak) 接頭辞で衝突を避ける。
STIMULUS_LABEL = "FkStimulus"    # 刺激要素
RESPONSE_LABEL = "FkResponse"    # 反応要素
MEANING_LABEL = "FkMeaning"      # 具体的意味づけ（開語彙）
CORE_LABEL = "FkCore"            # 中核評価（閉語彙・少数）
EPISODE_LABEL = "FkEpisode"      # 過去の出来事（エピソード記憶）

# --- 中核評価（閉集合・全体共有・固定シード） ---------------------------
# design_fear_structure.md の決定事項：この4種で開始（増減は後で可能）。
CORE_VALUATIONS = [
    {"code": "DANGER", "name": "危険"},
    {"code": "BAD", "name": "悪い・自分はダメ"},
    {"code": "POWERLESS", "name": "無力・逃げられない"},
    {"code": "UNENDING", "name": "終わらない"},
]
CORE_CODES = [c["code"] for c in CORE_VALUATIONS]
CORE_NAME = {c["code"]: c["name"] for c in CORE_VALUATIONS}

# --- 拡散活性（recall の中核）のパラメータ ------------------------------
SOURCE_TOP_K = 3          # クエリに一致させる Stimulus（活性の入口）の最大数
SOURCE_SIM_MIN = 0.20     # 入口とみなす最小コサイン類似度
SPREAD_HOPS = 3           # 恐怖構造をたどるホップ数
HOP_DECAY = 0.6           # 1 ホップごとの活性の減衰率
# 関係ごとの伝わりやすさ。忠実版オントロジーでは**一律 1.0**（2026-07-13 決定）。
# 理由: 連想強度は本来リンクごとに頻度で決まる可変量だが、それをエッジに載せる妥当な
# 方法が無い（頻度はfanと綱引き／cosは病的連想と逆符号）。よって「連想強度の差は
# モデル化しない」と限界宣言し、型による強弱付け(旧REL_WEIGHT)もやめて平坦化する。
REL_WEIGHT = {
    "EVOKES": 1.0,       # 刺激↔反応
    "MEANS": 1.0,        # 刺激/反応↔意味
    "CO_OCCURS": 1.0,    # 刺激↔刺激（同時性）
    "SIMILAR": 1.0,      # 意味的に近い刺激↔刺激（刺激般化）
    "RECALLS": 1.0,      # 刺激↔記憶（層間の橋）
    "BINDS": 1.0,        # 記憶↔反応/意味（束ね＝S-R-Mの束・層間の橋）
}

# --- 双方向伝搬（Foa & Kozak の連想ネットワークへの忠実化） --------------
# 恐怖構造の結合は「連想」であり向きを持たない（部分の活性化が構造全体を活性化する）。
# 忠実版では順逆同一(1.0)で対称に流す。方向の非対称はモデル化しない（限界として明記）。
BACKWARD_FLOW = 1.0

# --- 刺激般化（意味的に近い刺激を SIMILAR で結ぶ・汎用則） --------------
# ある刺激が、意味的に近い別の刺激の連想も呼び起こす（stimulus generalization）。
STIM_SIMILAR_MIN = 0.45   # SIMILAR を張る最小コサイン類似度
STIM_SIMILAR_TOPN = 3     # 1 刺激あたり結ぶ近傍数の上限

# --- 脅威方向への拡散バイアス（fear structure の threat bias・汎用則） --
# 「危険」等の中核に近いノードほど活性が乗りやすくする（中核からの距離ベースの重み付け）。
# 恐怖構造は脅威方向へ活性が流れやすい、という PTSD の特性を素直に表現する。
# 忠実版では撤去（DANGER_BIAS=0）。脅威中心性は外付けでなく構造から創発させる。
THREAT_CORES = ["DANGER"]     # （中核層は廃止したため現状は無効）
DANGER_BIAS = 0.0             # 0=無効（外付けの脅威バイアスは使わない）
DANGER_PULL_DECAY = 0.5       # （DANGER_BIAS=0 のとき未使用）

# --- 引き金ゲート（トラウマ反応 vs 平静の切り替え・一般性の担保） -------
# 恐怖構造が「本当に」強く発火したときだけ破局的発話に飛ぶ。そうでなければ平静に応答する。
# (a) 入口の刺激が十分に一致し(seedSim>=閾値) かつ (b) 想起が負の記憶優位、のとき引き金とみなす。
TRIGGER_SIM_MIN = 0.35        # 引き金とみなす最小の入口類似度（下回れば平静＝空振り）
TOP_NODES = 12            # 応答文脈に渡す活性ノード数
TOP_EPISODES = 4          # 応答文脈に渡す想起エピソード数

# --- 記憶構築（チャンク分割） -------------------------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
