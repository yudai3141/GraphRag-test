"""エージェント全体の設定。

ACT-R のパラメータ・モデル名・Neo4j のラベル名前空間などをここに集約する。
値を変えたいときは基本このファイルだけ見ればよい（可読性優先・DI なし）。
"""

import os

from dotenv import load_dotenv, find_dotenv

# --- パス ---------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# エピソード記憶の元になるテキスト（simulation のものを流用）
SOURCE_TEXT = os.path.join(_ROOT, "simulation", "ptsd_story.txt")


def load_env() -> None:
    """リポジトリルートの .env を読み込む（どの cwd からでも見つける）。"""
    load_dotenv(find_dotenv())


# --- モデル -------------------------------------------------------------
CHAT_MODEL = "gpt-5.4"            # 応答生成（simulation と同じ既定。実在 ID か要確認）
EXTRACT_MODEL = "gpt-5.4"         # エピソード抽出（構造化出力）
EMBED_MODEL = "text-embedding-3-small"

# --- ACT-R activation パラメータ（tmp-actr/run_final.py 準拠） -----------
DECAY = 0.1                       # ベースレベル減衰 d
RETRIEVAL_THRESHOLD = -2.0        # この値未満は想起しない
SPREADING_WEIGHT = 15.0           # 類似度（spreading）の重み
INSTANT_NOISE = 0.3               # 瞬時ノイズ s（ロジスティック）

# --- 二段想起（seed → 連想）の設定 -------------------------------------
SEED_K = 1                        # ACT-R で特定する中心エピソード数（まずは1）
RELATED_LIMIT = 3                 # seed から連想で引き出す関連エピソード数
ASSOC_HOPS = 1                    # 意味層（エンティティ）の連想ホップ数

# --- B(m) の初期割当（ランダム。将来は抽出時刻ベースに置換） -------------
BM_INIT_MEAN = 0.0
BM_INIT_STD = 0.5

# --- Neo4j のラベル名前空間（simulation の Document/__Entity__ と分離） ---
EPISODE_LABEL = "Episode"
ENTITY_LABEL = "MemEntity"

# --- 記憶構築（チャンク分割） -------------------------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
