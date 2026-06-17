"""Neo4j アクセス層（エピソード記憶グラフの CRUD）。

simulation と同じ Neo4j インスタンスを使うが、ラベルは Episode / MemEntity に分けて
名前空間を分離する（simulation の Document / __Entity__ とは衝突しない）。
"""

import os
import re

from neo4j import GraphDatabase

from .. import config
from ..domain.models import Episode, MemEntity


def _safe_rel(rel: str, default: str = "RELATED_TO") -> str:
    """関係名を Cypher の関係タイプに使える識別子へ正規化する。"""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", (rel or "").strip()).upper()
    return cleaned or default


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
    def clear_episodic_memory(self) -> None:
        """エピソード記憶（Episode / MemEntity）のみ削除。simulation グラフは温存。"""
        with self._session() as session:
            session.run(f"MATCH (n:{config.EPISODE_LABEL}) DETACH DELETE n")
            session.run(f"MATCH (n:{config.ENTITY_LABEL}) DETACH DELETE n")

    def add_episode(self, ep: Episode) -> None:
        ep_label, ent_label = config.EPISODE_LABEL, config.ENTITY_LABEL
        with self._session() as session:
            session.run(
                f"""
                MERGE (e:{ep_label} {{id: $id}})
                SET e.context = $context, e.event = $event, e.sensory = $sensory,
                    e.b_m = $b_m, e.t_created = $t_created,
                    e.presentations = $presentations, e.embedding = $embedding
                """,
                id=ep.id, context=ep.context, event=ep.event, sensory=ep.sensory,
                b_m=ep.b_m, t_created=ep.t_created,
                presentations=ep.presentations, embedding=ep.embedding,
            )
            for ent in ep.entities:
                rel = _safe_rel(ent.relation)
                session.run(
                    f"""
                    MERGE (ent:{ent_label} {{name: $name}})
                    SET ent.type = $type
                    WITH ent
                    MATCH (e:{ep_label} {{id: $eid}})
                    MERGE (e)-[:`{rel}`]->(ent)
                    """,
                    name=ent.name, type=ent.type, eid=ep.id,
                )

    def build_entity_associations(self) -> int:
        """意味層を構築する：同一エピソードに共起したエンティティ同士を ASSOC で結ぶ。

        重み weight は共起回数。既存の ASSOC は作り直す。戻り値は ASSOC 本数。
        """
        ep_label, ent_label = config.EPISODE_LABEL, config.ENTITY_LABEL
        with self._session() as session:
            session.run(f"MATCH (:{ent_label})-[r:ASSOC]-(:{ent_label}) DELETE r")
            session.run(
                f"""
                MATCH (e:{ep_label})-->(a:{ent_label}), (e)-->(b:{ent_label})
                WHERE a.name < b.name
                MERGE (a)-[r:ASSOC]->(b)
                ON CREATE SET r.weight = 1
                ON MATCH SET r.weight = r.weight + 1
                """
            )
            return session.run(
                f"MATCH (:{ent_label})-[r:ASSOC]-(:{ent_label}) RETURN count(r) AS c"
            ).single()["c"]

    def associative_episodes(self, seed_id: str, limit: int = 3) -> list[dict]:
        """seed エピソードのエンティティ（＋その ASSOC 連想先）を共有する別エピソードを返す。

        戻り値: [{"id": エピソードID, "shared": 共有エンティティ数}, ...]（shared 降順）。
        """
        ep_label, ent_label = config.EPISODE_LABEL, config.ENTITY_LABEL
        with self._session() as session:
            records = session.run(
                f"""
                MATCH (seed:{ep_label} {{id: $id}})-->(ent:{ent_label})
                OPTIONAL MATCH (ent)-[:ASSOC]-(rel:{ent_label})
                WITH seed, collect(DISTINCT ent.name) + collect(DISTINCT rel.name) AS names
                UNWIND names AS nm
                WITH seed, nm WHERE nm IS NOT NULL
                MATCH (other:{ep_label})-->(x:{ent_label} {{name: nm}})
                WHERE other <> seed
                WITH other, count(DISTINCT x) AS shared
                RETURN other.id AS id, shared
                ORDER BY shared DESC, other.t_created ASC
                LIMIT $limit
                """,
                id=seed_id, limit=limit,
            )
            return [{"id": r["id"], "shared": r["shared"]} for r in records]

    def link_next(self, prev_id: str, next_id: str) -> None:
        """エピソードを時系列で連結する（prev -[NEXT]-> next）。"""
        with self._session() as session:
            session.run(
                f"""
                MATCH (a:{config.EPISODE_LABEL} {{id: $a}}), (b:{config.EPISODE_LABEL} {{id: $b}})
                MERGE (a)-[:NEXT]->(b)
                """,
                a=prev_id, b=next_id,
            )

    # --- 読み込み（対話時） ---------------------------------------------
    def load_all_episodes(self) -> list[Episode]:
        with self._session() as session:
            records = session.run(
                f"""
                MATCH (e:{config.EPISODE_LABEL})
                OPTIONAL MATCH (e)-[r]->(ent:{config.ENTITY_LABEL})
                WITH e, collect(
                    CASE WHEN ent IS NULL THEN NULL
                    ELSE {{name: ent.name, type: ent.type, relation: type(r)}} END
                ) AS ents
                RETURN e{{.*}} AS e, ents
                ORDER BY e.t_created
                """
            )
            episodes = []
            for rec in records:
                e = rec["e"]
                ents = [MemEntity(**d) for d in rec["ents"] if d is not None]
                episodes.append(
                    Episode(
                        id=e["id"],
                        context=e.get("context", ""),
                        event=e.get("event", ""),
                        sensory=e.get("sensory", ""),
                        b_m=e.get("b_m", 0.0),
                        t_created=e.get("t_created", 0.0),
                        presentations=list(e.get("presentations") or []),
                        embedding=list(e.get("embedding") or []),
                        entities=ents,
                    )
                )
            return episodes

    # --- 更新（想起による強化） -----------------------------------------
    def reinforce(self, episode_id: str, t: float) -> None:
        """想起された時刻 t を presentations に追記し、ベースレベルを強化する。"""
        with self._session() as session:
            session.run(
                f"""
                MATCH (e:{config.EPISODE_LABEL} {{id: $id}})
                SET e.presentations = coalesce(e.presentations, []) + [$t]
                """,
                id=episode_id, t=t,
            )
