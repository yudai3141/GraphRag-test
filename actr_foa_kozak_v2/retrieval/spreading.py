"""恐怖構造上の拡散活性（v2 の想起の中核）。

v1 の「ACT-R で seed エピソードを1つ選ぶ」方式と違い、ここでは:

  ① クエリ埋め込みに近い刺激(Stimulus)を活性の入口にする（クエリはノード化しない）
  ② 恐怖構造のエッジ（EVOKES/MEANS/ROLLS_UP/RECALLS/LEADS_TO/CO_OCCURS）に沿って活性を拡散
     ・1 ホップごとに HOP_DECAY で減衰
     ・出力エッジで活性を分割（fan effect）し、関係ごとの重み REL_WEIGHT を掛ける
     ・中核評価へ収束する構造・LEADS_TO(負→負) がそのまま「PTSDらしい連想」を生む
  ③ 活性化した刺激・反応・意味・中核評価・過去エピソードを降順に集める

良い記憶（回復エピソード等）は恐怖構造へのエッジを持たない＝活性が届かず想起されない。
"""

from typing import Dict, List

import numpy as np

from .. import config
from ..domain.models import (
    ActivatedEpisode, ActivatedNode, Episode, FearGraph, FearNode, RecallResult,
)
from ..infrastructure.embeddings import EmbeddingService, cosine


class SpreadingActivation:
    def __init__(
        self,
        embedder: EmbeddingService,
        hops: int = config.SPREAD_HOPS,
        hop_decay: float = config.HOP_DECAY,
        rel_weight: Dict[str, float] | None = None,
        source_top_k: int = config.SOURCE_TOP_K,
        source_sim_min: float = config.SOURCE_SIM_MIN,
    ):
        self.embedder = embedder
        self.hops = hops
        self.hop_decay = hop_decay
        self.rel_weight = rel_weight or config.REL_WEIGHT
        self.source_top_k = source_top_k
        self.source_sim_min = source_sim_min

    def recall(self, query: str, graph: FearGraph) -> RecallResult:
        by_key: Dict[str, FearNode] = {n.key: n for n in graph.nodes}
        ep_by_key: Dict[str, Episode] = {e.key: e for e in graph.episodes}

        # 出力方向の隣接リスト: src -> [(dst, rel, weight), ...]
        adj: Dict[str, List[tuple]] = {}
        for e in graph.edges:
            adj.setdefault(e.src, []).append((e.dst, e.rel, e.weight))

        # ① 入口：クエリに近い刺激ノード
        qv = self.embedder.embed(query)
        sims = []
        for s in graph.stimuli():
            if not s.embedding:
                continue
            sim = cosine(qv, np.asarray(s.embedding, dtype=np.float32))
            sims.append((s, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        seeds = [(s, sim) for s, sim in sims[: self.source_top_k] if sim >= self.source_sim_min]

        activation: Dict[str, float] = {}
        reached_hop: Dict[str, int] = {}
        for s, sim in seeds:
            activation[s.key] = activation.get(s.key, 0.0) + sim
            reached_hop.setdefault(s.key, 0)

        # ② 拡散：ホップごとに活性を流す
        frontier = dict(activation)
        for hop in range(1, self.hops + 1):
            nxt: Dict[str, float] = {}
            for src_key, a in frontier.items():
                outs = adj.get(src_key)
                if not outs:
                    continue
                total_w = sum(w for _, _, w in outs) or 1.0
                for dst_key, rel, w in outs:
                    share = w / total_w                      # fan effect（出力で分割）
                    flow = a * self.hop_decay * self.rel_weight.get(rel, 1.0) * share
                    if flow <= 1e-6:
                        continue
                    nxt[dst_key] = nxt.get(dst_key, 0.0) + flow
            for k, v in nxt.items():
                activation[k] = activation.get(k, 0.0) + v
                reached_hop.setdefault(k, hop)
            frontier = nxt
            if not frontier:
                break

        # ③ 集計：恐怖構造ノードとエピソードに分ける
        act_nodes: List[ActivatedNode] = []
        act_eps: List[ActivatedEpisode] = []
        core_activation: Dict[str, float] = {}
        seed_keys = {s.key for s, _ in seeds}

        for key, a in activation.items():
            if key in by_key:
                node = by_key[key]
                if node.label == config.CORE_LABEL:
                    core_activation[node.core_code or node.name] = a
                if key in seed_keys:
                    continue  # 入口は seeds 側で持つ
                act_nodes.append(ActivatedNode(node=node, activation=a, hop=reached_hop.get(key, 0)))
            elif key in ep_by_key:
                act_eps.append(ActivatedEpisode(
                    episode=ep_by_key[key], activation=a, hop=reached_hop.get(key, 0)))

        act_nodes.sort(key=lambda x: x.activation, reverse=True)
        act_eps.sort(key=lambda x: x.activation, reverse=True)

        seed_nodes = [
            ActivatedNode(node=s, activation=activation[s.key], hop=0) for s, _ in seeds
        ]

        return RecallResult(
            query=query,
            seeds=seed_nodes,
            nodes=act_nodes[: config.TOP_NODES],
            episodes=act_eps[: config.TOP_EPISODES],
            core_activation=core_activation,
        )
