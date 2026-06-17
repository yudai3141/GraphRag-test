"""二段想起：ACT-R で seed を特定し、意味層の連想で関連エピソードを集める。

  ① ActivationEngine で全エピソードを採点 → 最も活性の高い seed を1つ特定（エピソード記憶）
  ② seed のエンティティ（＋ ASSOC 連想先）を共有する別エピソードを取得（意味記憶の連想）

これは ACT-R の spreading activation（手がかり中のエンティティ=source から、それを含む
チャンク=エピソードへ活性が流れる）を、エピソード層と意味層に分けて素直に表現したもの。
"""

from typing import List

from .. import config
from ..domain.models import AssociatedEpisode, Episode, RecallResult
from ..infrastructure.neo4j_store import Neo4jStore
from .base import ActivationEngine


class MemoryRecaller:
    def __init__(
        self,
        engine: ActivationEngine,
        store: Neo4jStore,
        related_limit: int = config.RELATED_LIMIT,
    ):
        self.engine = engine
        self.store = store
        self.related_limit = related_limit

    def recall(self, query: str, episodes: List[Episode], now: float) -> RecallResult:
        by_id = {ep.id: ep for ep in episodes}

        # ① ACT-R activation で seed を特定
        candidates = self.engine.rank(query, episodes, now)
        seed = candidates[0] if candidates and candidates[0].above_threshold else None

        # ② seed のエンティティ連想で関連エピソードを取得
        related: List[AssociatedEpisode] = []
        if seed is not None:
            rows = self.store.associative_episodes(seed.episode.id, limit=self.related_limit)
            for row in rows:
                ep = by_id.get(row["id"])
                if ep is not None:
                    related.append(AssociatedEpisode(episode=ep, shared=row["shared"]))

        return RecallResult(candidates=candidates, seed=seed, related=related)
