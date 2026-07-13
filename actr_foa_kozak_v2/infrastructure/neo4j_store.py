"""Neo4j アクセス層（恐怖構造グラフの CRUD）。

v1/simulation と同じ Neo4j インスタンスを使うが、ラベルはすべて "Fk" 接頭辞
（FkStimulus / FkResponse / FkMeaning / FkCore / FkEpisode）で名前空間を分離する。

ノードは name をキーにする（中核評価 FkCore は code を name に入れ、日本語名は jp に持つ）。
エッジは MERGE で作り、既出なら weight を加算する（共起の強さ＝結合の強さ）。
"""

import os

from neo4j import GraphDatabase

from .. import config
from ..domain.models import Edge, Episode, FearGraph, FearNode, node_key

# 恐怖構造のノードラベル一覧（読み込み・削除に使う）
_FK_LABELS = [
    config.STIMULUS_LABEL,
    config.RESPONSE_LABEL,
    config.MEANING_LABEL,
    config.CORE_LABEL,
    config.EPISODE_LABEL,
]


class Neo4jStore:
    def __init__(self):
        self._driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        self._db = os.environ.get("NEO4J_DATABASE", "neo4j")

    def _session(self):
        return self._driver.session(database=self._db)

    def close(self) -> None:
        self._driver.close()

    # --- 書き込み（記憶構築） -------------------------------------------
    def clear_fear_graph(self) -> None:
        """恐怖構造（Fk* ラベル）のみ削除。v1/simulation のグラフは温存。"""
        with self._session() as s:
            for label in _FK_LABELS:
                s.run(f"MATCH (n:{label}) DETACH DELETE n")

    def ensure_core_valuations(self) -> None:
        """中核評価ノード（固定4種）を用意する。"""
        with self._session() as s:
            for c in config.CORE_VALUATIONS:
                s.run(
                    f"MERGE (n:{config.CORE_LABEL} {{name: $code}}) SET n.jp = $jp",
                    code=c["code"], jp=c["name"],
                )

    def merge_node(self, label: str, name: str, embedding: list | None = None) -> None:
        """名前をキーにノードを upsert（あれば埋め込みだけ更新）。"""
        with self._session() as s:
            s.run(
                f"MERGE (n:{label} {{name: $name}}) "
                "SET n.embedding = coalesce($embedding, n.embedding)",
                name=name, embedding=embedding,
            )

    def merge_edge(
        self, src_label: str, src_name: str, rel: str,
        dst_label: str, dst_name: str, weight_inc: float = 1.0,
    ) -> None:
        """ノード間の結合を MERGE。既出なら weight を加算する。"""
        with self._session() as s:
            s.run(
                f"""
                MATCH (a:{src_label} {{name: $a}}), (b:{dst_label} {{name: $b}})
                MERGE (a)-[r:`{rel}`]->(b)
                ON CREATE SET r.weight = $w
                ON MATCH SET r.weight = coalesce(r.weight, 0) + $w
                """,
                a=src_name, b=dst_name, w=weight_inc,
            )

    def add_episode(self, ep: Episode) -> None:
        with self._session() as s:
            s.run(
                f"""
                MERGE (e:{config.EPISODE_LABEL} {{name: $id}})
                SET e.context = $context, e.event = $event, e.sensory = $sensory,
                    e.valence = $valence, e.t_created = $t_created, e.embedding = $embedding
                """,
                id=ep.id, context=ep.context, event=ep.event, sensory=ep.sensory,
                valence=ep.valence, t_created=ep.t_created, embedding=ep.embedding,
            )

    def merge_recalls(self, stimulus_name: str, episode_id: str, weight_inc: float = 1.0) -> None:
        """現在の刺激 → 過去のエピソードを想起する結合（RECALLS）。"""
        self.merge_edge(
            config.STIMULUS_LABEL, stimulus_name, "RECALLS",
            config.EPISODE_LABEL, episode_id, weight_inc,
        )

    def merge_binds(self, episode_id: str, dst_label: str, dst_name: str,
                    weight_inc: float = 1.0) -> None:
        """記憶＝束ねられた刺激-反応-意味（BINDS）。エピソード↔反応/意味を束ねる。

        Lang/Foa の「トラウマ記憶は刺激・反応・意味が束ねられた構造」に対応。
        恐怖構造レイヤとエピソード記憶レイヤをつなぐ層間の橋（RECALLS と対）。
        """
        self.merge_edge(
            config.EPISODE_LABEL, episode_id, "BINDS",
            dst_label, dst_name, weight_inc,
        )

    def merge_cooccurs(self, name_a: str, name_b: str, weight_inc: float = 1.0) -> None:
        """刺激どうしの同時性（CO_OCCURS、無向として片方向で保持）。"""
        lo, hi = sorted([name_a, name_b])
        self.merge_edge(
            config.STIMULUS_LABEL, lo, "CO_OCCURS",
            config.STIMULUS_LABEL, hi, weight_inc,
        )

    def merge_similar(self, name_a: str, name_b: str, sim: float) -> None:
        """意味的に近い刺激どうしを SIMILAR で結ぶ（刺激般化。重み=類似度、冪等）。"""
        lo, hi = sorted([name_a, name_b])
        with self._session() as s:
            s.run(
                f"""
                MATCH (a:{config.STIMULUS_LABEL} {{name: $a}}), (b:{config.STIMULUS_LABEL} {{name: $b}})
                MERGE (a)-[r:SIMILAR]->(b) SET r.weight = $w
                """,
                a=lo, b=hi, w=sim,
            )

    # --- 読み込み（対話時） ---------------------------------------------
    def load_fear_graph(self) -> FearGraph:
        """恐怖構造グラフ全体をメモリに読み込む（拡散活性の入力）。"""
        with self._session() as s:
            # ノード（刺激/反応/具体的意味/中核評価）
            nodes: list[FearNode] = []
            node_labels = [
                config.STIMULUS_LABEL, config.RESPONSE_LABEL,
                config.MEANING_LABEL, config.CORE_LABEL,
            ]
            for label in node_labels:
                for r in s.run(
                    f"MATCH (n:{label}) RETURN n.name AS name, n.embedding AS emb, n.jp AS jp"
                ):
                    nodes.append(FearNode(
                        key=node_key(label, r["name"]),
                        label=label,
                        name=r["jp"] if (label == config.CORE_LABEL and r["jp"]) else r["name"],
                        core_code=r["name"] if label == config.CORE_LABEL else None,
                        embedding=list(r["emb"] or []),
                    ))

            # エピソード
            episodes: list[Episode] = []
            for r in s.run(
                f"MATCH (e:{config.EPISODE_LABEL}) RETURN e{{.*}} AS e ORDER BY e.t_created"
            ):
                e = r["e"]
                episodes.append(Episode(
                    id=e["name"],
                    context=e.get("context", ""),
                    event=e.get("event", ""),
                    sensory=e.get("sensory", ""),
                    valence=e.get("valence", "negative"),
                    t_created=e.get("t_created", 0.0),
                    embedding=list(e.get("embedding") or []),
                ))

            # エッジ（Fk* 間のみ）。キーは "ラベル::name"。
            edges: list[Edge] = []
            for r in s.run(
                """
                MATCH (a)-[r]->(b)
                WHERE any(l IN labels(a) WHERE l STARTS WITH 'Fk')
                  AND any(l IN labels(b) WHERE l STARTS WITH 'Fk')
                RETURN [l IN labels(a) WHERE l STARTS WITH 'Fk'][0] AS al, a.name AS an,
                       type(r) AS rel, coalesce(r.weight, 1.0) AS w,
                       [l IN labels(b) WHERE l STARTS WITH 'Fk'][0] AS bl, b.name AS bn
                """
            ):
                edges.append(Edge(
                    src=node_key(r["al"], r["an"]),
                    dst=node_key(r["bl"], r["bn"]),
                    rel=r["rel"],
                    weight=float(r["w"]),
                ))

        return FearGraph(nodes=nodes, episodes=episodes, edges=edges)

    def counts(self) -> dict:
        """構築結果の件数（確認用）。ノードはラベル別、エッジは関係型別。"""
        out = {}
        with self._session() as s:
            for label in _FK_LABELS:
                out[label] = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            for rel in config.REL_WEIGHT:
                out[rel] = s.run(
                    f"MATCH ()-[r:`{rel}`]->() RETURN count(r) AS c"
                ).single()["c"]
        return out
