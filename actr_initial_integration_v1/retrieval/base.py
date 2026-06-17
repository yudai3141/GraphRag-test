"""検索エンジンのインターフェース。

v1 では activation.py の自前実装を使う。将来 pyactr バックエンドに差し替える場合も、
この `ActivationEngine` を実装すれば agent 層を変えずに入れ替えられる。
"""

from abc import ABC, abstractmethod
from typing import List

from ..domain.models import Episode, RetrievalCandidate


class ActivationEngine(ABC):
    @abstractmethod
    def rank(
        self, query: str, episodes: List[Episode], now: float
    ) -> List[RetrievalCandidate]:
        """発話 query に対し、各エピソードの活性化 A(m) を計算して降順に返す。

        Args:
            query: ユーザー発話（＋必要なら直近文脈）。
            episodes: 対象のエピソード記憶。
            now: 現在の論理時刻（ベースレベルの時間減衰に使う）。
        """
        raise NotImplementedError
