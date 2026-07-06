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
# 関係ごとの伝わりやすさ。負→負(LEADS_TO)と想起(RECALLS)を強めに（PTSDらしさ）。
REL_WEIGHT = {
    "EVOKES": 1.0,        # 刺激→反応
    "MEANS": 1.0,         # 刺激/反応→意味
    "ROLLS_UP": 1.2,      # 具体的意味→中核評価（収束を効かせる）
    "RECALLS": 1.2,       # 現在の刺激→過去の記憶
    "LEADS_TO": 1.5,      # 嫌な記憶→嫌な記憶
    "CO_OCCURS": 0.7,     # 刺激どうしの同時性
}
TOP_NODES = 12            # 応答文脈に渡す活性ノード数
TOP_EPISODES = 4          # 応答文脈に渡す想起エピソード数

# --- 記憶構築（チャンク分割） -------------------------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
