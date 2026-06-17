"""ドメインモデル。

エピソード記憶（エピソード記憶＝自分が体験した出来事）を表すデータ構造。
ACT-R のチャンクに対応し、C-rep / event / S-rep の三表現を持つ。
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class MemEntity(BaseModel):
    """エピソードに登場する型付きエンティティ。"""

    name: str                  # 例: "加害者", "電車", "恐怖"
    type: str                  # 例: Person / Place / Emotion / Symptom / Treatment / Time / Concept
    relation: str              # エピソードからの関係: INVOLVES / AT / FELT / ...


class Episode(BaseModel):
    """エピソード記憶チャンク。

    - context : C-rep（状況・文脈。どこで・どんな状況だったか）
    - event   : 中核イベント（何が起きたか）
    - sensory : S-rep（感覚・感情）
    - b_m     : ベースレベルの初期バイアス（最初はランダム割当）
    - t_created    : 生成（記銘）時の論理時刻
    - presentations: 提示（記銘・想起）が起きた時刻列。想起のたびに追記され強化される
    - embedding    : エピソード本文の埋め込みベクトル（事前計算して保持）
    """

    id: str
    context: str
    event: str
    sensory: str
    b_m: float
    t_created: float
    presentations: List[float] = Field(default_factory=list)
    embedding: List[float] = Field(default_factory=list)
    entities: List[MemEntity] = Field(default_factory=list)

    def as_text(self) -> str:
        """埋め込み・表示用に三表現を1つのテキストへ。"""
        return f"状況: {self.context} / 出来事: {self.event} / 感覚: {self.sensory}"


class RetrievalCandidate(BaseModel):
    """ある発話に対する 1 エピソードの活性化計算結果（DEBUG 出力用に内訳を保持）。"""

    episode: Episode
    b_level: float             # B(m): ベースレベル + 時間減衰（+ b_m バイアス）
    similarity: float          # クエリとのコサイン類似度
    a_spreading: float         # 類似度 × 重み
    noise: float               # 瞬時ノイズ
    total_a: float             # A(m) = b_level + a_spreading + noise
    above_threshold: bool


class AssociatedEpisode(BaseModel):
    """seed から連想（共有エンティティ）で引き出された関連エピソード。"""

    episode: Episode
    shared: int                # seed 側エンティティ／その連想先と共有した数（多いほど近い）


class RecallResult(BaseModel):
    """1 ターンの想起結果（二段構成）。

    - candidates: 全エピソードの活性化ランキング（DEBUG 用）
    - seed      : ACT-R activation で最初に想起した中心エピソード（閾値未満なら None）
    - related   : seed のエンティティ連想で引き出した関連エピソード
    """

    candidates: List[RetrievalCandidate]
    seed: Optional[RetrievalCandidate] = None
    related: List[AssociatedEpisode] = Field(default_factory=list)
