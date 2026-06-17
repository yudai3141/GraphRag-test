"""ACT-R activation の自前実装。

tmp-actr/run_final.py の式に準拠:

    A(m) = B(m) + Σ (類似度 × spreading重み) + 瞬時ノイズ

  - B(m) : ベースレベル学習（時間減衰）。 B = ln(Σ_k (t_now - t_k)^(-d))
           さらに初期バイアス b_m を log-sum-exp で合成する（run_final と同じ）。
  - 類似度: クエリ埋め込みとエピソード埋め込みのコサイン類似度（embedding spreading）。
  - ノイズ: ロジスティック分布の瞬時ノイズ。
  retrieval_threshold 未満は想起対象外とし、A(m) の降順で返す。
"""

import math
import random
from typing import List

import numpy as np

from .. import config
from ..domain.models import Episode, RetrievalCandidate
from ..infrastructure.embeddings import EmbeddingService, cosine
from .base import ActivationEngine


def base_level(times: List[float], now: float, decay: float, b_m: float) -> float:
    """ベースレベル B(m)：時間減衰 + 初期バイアス b_m。"""
    diffs = [now - t for t in times if now - t > 0]
    if diffs:
        try:
            base = math.log(sum(d ** (-decay) for d in diffs))
        except (ValueError, OverflowError):
            base = -10.0
    else:
        base = -10.0
    # 初期バイアス b_m を合成（run_final.py と同じ log-sum-exp）
    try:
        return math.log(math.exp(base) + math.exp(b_m))
    except OverflowError:
        return max(base, b_m)


def instantaneous_noise(s: float) -> float:
    """ロジスティック分布に従う瞬時ノイズ（平均 0）。"""
    if s <= 0:
        return 0.0
    p = random.uniform(1e-9, 1 - 1e-9)
    return s * math.log((1 - p) / p)


class ActrActivationEngine(ActivationEngine):
    def __init__(
        self,
        embedder: EmbeddingService,
        decay: float = config.DECAY,
        threshold: float = config.RETRIEVAL_THRESHOLD,
        spreading_weight: float = config.SPREADING_WEIGHT,
        noise: float = config.INSTANT_NOISE,
    ):
        self.embedder = embedder
        self.decay = decay
        self.threshold = threshold
        self.spreading_weight = spreading_weight
        self.noise = noise

    def rank(
        self, query: str, episodes: List[Episode], now: float
    ) -> List[RetrievalCandidate]:
        query_vec = self.embedder.embed(query)
        candidates: List[RetrievalCandidate] = []
        for ep in episodes:
            times = ep.presentations or [ep.t_created]
            b_level = base_level(times, now, self.decay, ep.b_m)

            ep_vec = np.asarray(ep.embedding, dtype=np.float32)
            similarity = cosine(query_vec, ep_vec)
            a_spreading = similarity * self.spreading_weight

            noise = instantaneous_noise(self.noise)
            total_a = b_level + a_spreading + noise

            candidates.append(
                RetrievalCandidate(
                    episode=ep,
                    b_level=b_level,
                    similarity=similarity,
                    a_spreading=a_spreading,
                    noise=noise,
                    total_a=total_a,
                    above_threshold=total_a >= self.threshold,
                )
            )
        candidates.sort(key=lambda c: c.total_a, reverse=True)
        return candidates
