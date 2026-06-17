"""埋め込みサービス。

OpenAI の埋め込みでテキストをベクトル化する。同じテキストは使い回せるよう
インメモリでキャッシュする。
"""

import numpy as np
from langchain_openai import OpenAIEmbeddings

from .. import config


class EmbeddingService:
    def __init__(self, model: str = config.EMBED_MODEL):
        self._embedder = OpenAIEmbeddings(model=model)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            vec = self._embedder.embed_query(text)
            self._cache[text] = np.asarray(vec, dtype=np.float32)
        return self._cache[text]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度（0..1 にクリップ。負値は 0 に丸める）。"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(max(0.0, np.dot(a, b) / (na * nb)))
