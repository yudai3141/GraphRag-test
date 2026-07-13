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

from collections import deque
from typing import Dict, List

import numpy as np

from .. import config
from ..domain.models import (
    ActivatedEpisode, ActivatedNode, Episode, FearGraph, FearNode, RecallResult, node_key,
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
        danger_bias: float = config.DANGER_BIAS,
        danger_decay: float = config.DANGER_PULL_DECAY,
        threat_cores: List[str] | None = None,
        backward: float = config.BACKWARD_FLOW,
    ):
        self.embedder = embedder
        self.hops = hops
        self.hop_decay = hop_decay
        self.rel_weight = rel_weight or config.REL_WEIGHT
        self.source_top_k = source_top_k
        self.source_sim_min = source_sim_min
        self.danger_bias = danger_bias
        self.danger_decay = danger_decay
        self.threat_cores = threat_cores or config.THREAT_CORES
        # 結合は連想（無向）とみなし、逆向きにも backward 倍で伝搬させる（0=一方向）。
        # Foa & Kozak: 構造の一部の活性化は全体に波及する。危険ハブから
        # トラウマ記憶側へ「戻る」流れ（侵入想起）はこれが担う。
        self.backward = backward

    def _danger_pull(self, graph: FearGraph, adj: Dict[str, List[tuple]]) -> Dict[str, float]:
        """各ノードの「脅威中核への近さ」pull ∈ (0,1] を返す（threat bias 用）。

        中核（DANGER 等）から辺を逆向きに BFS し、中核までのホップ数 d を測る。
        pull = danger_decay ^ d（近いほど大きい）。到達しないノードは 0（バイアスなし）。
        """
        if self.danger_bias <= 0:
            return {}
        node_keys = {n.key for n in graph.nodes}
        radj: Dict[str, List[str]] = {}
        for src, outs in adj.items():
            for dst, _rel, _w in outs:
                radj.setdefault(dst, []).append(src)
        starts = [node_key(config.CORE_LABEL, c) for c in self.threat_cores]
        dist: Dict[str, int] = {k: 0 for k in starts if k in node_keys}
        dq = deque(dist)
        while dq:
            u = dq.popleft()
            for v in radj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        return {k: self.danger_decay ** d for k, d in dist.items()}

    def compute(self, query: str, graph: FearGraph):
        """拡散活性の生データを返す（可視化などで共用）。

        戻り値: (activation: key->活性値, reached_hop: key->到達ホップ, seeds: [(FearNode, 類似度)])
        """
        # 隣接リスト: src -> [(dst, rel, weight), ...]
        # backward > 0 のときは逆向きの連想も張る（重みは backward 倍）。
        adj: Dict[str, List[tuple]] = {}
        for e in graph.edges:
            adj.setdefault(e.src, []).append((e.dst, e.rel, e.weight))
            if self.backward > 0:
                adj.setdefault(e.dst, []).append((e.src, e.rel, e.weight * self.backward))

        # 脅威方向バイアス：各ノードの「危険中核への近さ」を先に測っておく
        pull = self._danger_pull(graph, adj)

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
        # 双方向時の往復（A→B→A の即時ピンポン）は、直前ホップで自分に
        # 流し込んできた相手へは返さない、というルールで抑える。
        frontier = dict(activation)
        prev_contrib: Dict[str, set] = {}
        for hop in range(1, self.hops + 1):
            nxt: Dict[str, float] = {}
            contrib: Dict[str, set] = {}
            for src_key, a in frontier.items():
                outs = adj.get(src_key)
                if not outs:
                    continue
                total_w = sum(w for _, _, w in outs) or 1.0
                back_to = prev_contrib.get(src_key, ())
                for dst_key, rel, w in outs:
                    if dst_key in back_to:                   # 即時の逆流は返さない
                        continue
                    share = w / total_w                      # fan effect（出力で分割）
                    # 脅威方向バイアス：危険中核に近い到達先ほど活性が乗りやすい
                    threat = 1.0 + self.danger_bias * pull.get(dst_key, 0.0)
                    flow = a * self.hop_decay * self.rel_weight.get(rel, 1.0) * share * threat
                    if flow <= 1e-6:
                        continue
                    nxt[dst_key] = nxt.get(dst_key, 0.0) + flow
                    contrib.setdefault(dst_key, set()).add(src_key)
            for k, v in nxt.items():
                activation[k] = activation.get(k, 0.0) + v
                reached_hop.setdefault(k, hop)
            frontier = nxt
            prev_contrib = contrib
            if not frontier:
                break

        return activation, reached_hop, seeds

    def recall(self, query: str, graph: FearGraph) -> RecallResult:
        by_key: Dict[str, FearNode] = {n.key: n for n in graph.nodes}
        ep_by_key: Dict[str, Episode] = {e.key: e for e in graph.episodes}

        activation, reached_hop, seeds = self.compute(query, graph)

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

        # seed の activation は入口の一致度（クエリ類似度）を保持する（引き金ゲート判定に使う）
        seed_nodes = [
            ActivatedNode(node=s, activation=sim, hop=0) for s, sim in seeds
        ]

        return RecallResult(
            query=query,
            seeds=seed_nodes,
            nodes=act_nodes[: config.TOP_NODES],
            episodes=act_eps[: config.TOP_EPISODES],
            core_activation=core_activation,
        )
