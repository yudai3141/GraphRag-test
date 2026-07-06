"""ドメインモデル（v2: 恐怖構造）。

Foa & Kozak の恐怖構造を、純粋なデータ構造として表す（外部依存なし）。
- FearNode : 刺激 / 反応 / 具体的意味づけ / 中核評価 のノード
- Episode  : 過去に体験した出来事（エピソード記憶）
- Edge     : ノード間の結合（EVOKES / MEANS / ROLLS_UP / RECALLS / LEADS_TO / CO_OCCURS）
- FearGraph: 上記をまとめてメモリに載せた恐怖構造グラフ（拡散活性の入力）
- 想起結果（ActivatedNode / ActivatedEpisode / RecallResult）
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def node_key(label: str, name: str) -> str:
    """ラベル＋名前で一意なノードキー（役割違いの同名を区別する）。"""
    return f"{label}::{name}"


class FearNode(BaseModel):
    """恐怖構造のノード（刺激 / 反応 / 具体的意味づけ / 中核評価）。"""

    key: str                          # node_key(label, name)
    label: str                        # FkStimulus / FkResponse / FkMeaning / FkCore
    name: str                         # 表示名（中核評価は日本語名、code は core_code）
    core_code: Optional[str] = None   # FkMeaning がどの中核へ収束するか / FkCore の code
    embedding: List[float] = Field(default_factory=list)


class Episode(BaseModel):
    """過去に体験した出来事。"""

    id: str
    context: str
    event: str
    sensory: str
    valence: str = "negative"         # negative / neutral / positive（連想の可否に効く）
    t_created: float = 0.0
    embedding: List[float] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"FkEpisode::{self.id}"

    def as_text(self) -> str:
        return f"状況: {self.context} / 出来事: {self.event} / 感覚: {self.sensory}"


class Edge(BaseModel):
    """ノード間の結合（キー参照）。weight は結合の強さ。"""

    src: str                          # ノードキー
    dst: str                          # ノードキー
    rel: str                          # EVOKES / MEANS / ROLLS_UP / RECALLS / LEADS_TO / CO_OCCURS
    weight: float = 1.0


class FearGraph(BaseModel):
    """メモリ上の恐怖構造グラフ。拡散活性はこれを入力に計算する。"""

    nodes: List[FearNode] = Field(default_factory=list)
    episodes: List[Episode] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    def stimuli(self) -> List[FearNode]:
        return [n for n in self.nodes if n.label == "FkStimulus"]


class ActivatedNode(BaseModel):
    """拡散活性で活性化した恐怖構造ノード（DEBUG 用に内訳を保持）。"""

    node: FearNode
    activation: float
    hop: int                          # 何ホップ目で最初に到達したか（0=入口）


class ActivatedEpisode(BaseModel):
    """拡散活性で想起された過去エピソード。"""

    episode: Episode
    activation: float
    hop: int


class RecallResult(BaseModel):
    """1 ターンの想起結果（拡散活性版）。"""

    query: str
    seeds: List[ActivatedNode] = Field(default_factory=list)         # 入口になった Stimulus
    nodes: List[ActivatedNode] = Field(default_factory=list)         # 活性ノード（降順）
    episodes: List[ActivatedEpisode] = Field(default_factory=list)   # 想起エピソード（降順）
    core_activation: Dict[str, float] = Field(default_factory=dict)  # 中核評価 code -> 活性合計
