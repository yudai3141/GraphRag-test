"""二段想起の「流れ」を1例で可視化する（pyvis 階層レイアウト）。

  クエリ → ① seed想起 → ② エンティティ抽出 → ③ ASSOC連想 → ④ 関連エピソード想起 → ⑤ 応答

実データを1クエリ分流して、各段を左→右の段（level）に並べた図を flow.html に書き出す。
記憶は変更しない（reinforce しない読み取り専用）。

  uv run --extra viz python -m actr_initial_integration_v1.flow_viz "電車に乗るのは平気ですか？"
"""

import os
import sys

from . import config
from .prompts import templates
from .infrastructure.embeddings import EmbeddingService
from .infrastructure.llm import LLMClient
from .infrastructure.neo4j_store import Neo4jStore
from .retrieval.activation import ActrActivationEngine
from .retrieval.recall import MemoryRecaller

# 段ごとの色
C_QUERY, C_EP, C_ENT, C_AENT, C_RESP = "#f58231", "#4363d8", "#e6194B", "#f032e6", "#3cb44b"

_LEGEND = (
    '<div style="position:fixed;top:10px;right:10px;z-index:1000;background:rgba(255,255,255,.96);'
    'border:1px solid #ccc;border-radius:8px;padding:10px 12px;font:12px sans-serif;line-height:1.6;">'
    '<b>二段想起の流れ</b><br>'
    f'<span style="color:{C_QUERY}">●</span> ① クエリ<br>'
    f'<span style="color:{C_EP}">●</span> ② seedエピソード（ACT-R想起）<br>'
    f'<span style="color:{C_ENT}">●</span> ③ seedのエンティティ<br>'
    f'<span style="color:{C_AENT}">●</span> ④ ASSOC連想で想起したエンティティ<br>'
    f'<span style="color:{C_EP}">●</span> ⑤ 連想で引いた関連エピソード<br>'
    f'<span style="color:{C_RESP}">●</span> ⑥ 応答</div>'
)

_OPTIONS = """
{
  "layout": {"hierarchical": {"enabled": true, "direction": "LR",
      "sortMethod": "directed", "levelSeparation": 280, "nodeSpacing": 110}},
  "physics": {"enabled": false},
  "edges": {"smooth": {"type": "cubicBezier", "forceDirection": "horizontal"}}
}
"""


def export_flow(query: str, output_path: str = "flow.html") -> str:
    from pyvis.network import Network

    embedder = EmbeddingService()
    engine = ActrActivationEngine(embedder)
    store = Neo4jStore()
    recaller = MemoryRecaller(engine, store)
    llm = LLMClient()

    episodes = store.load_all_episodes()
    by_id = {e.id: e for e in episodes}
    now = max((e.t_created for e in episodes), default=0.0) + 2.0

    # ①〜④：想起
    result = recaller.recall(query, episodes, now)
    seed = result.seed
    related = result.related

    # ⑥：応答（読み取り専用＝reinforce しない）
    seed_ep = seed.episode if seed else None
    related_eps = [a.episode for a in related]
    response = llm.complete(
        templates.SYSTEM, templates.build_user_prompt(query, seed_ep, related_eps)
    )

    # 連想エンティティと「どのエンティティ経由で関連エピソードに繋がったか」を取得
    seed_names = [e.name for e in seed_ep.entities] if seed_ep else []
    related_ids = [e.id for e in related_eps]
    assoc_nbrs: dict[str, list[str]] = {}     # seed entity -> ASSOC 近傍
    ep_ents: dict[str, list[str]] = {}        # related ep id -> その entity 名
    if seed_names and related_ids:
        with store._session() as s:
            for r in s.run(
                f"MATCH (a:{config.ENTITY_LABEL})-[:ASSOC]-(b:{config.ENTITY_LABEL}) "
                "WHERE a.name IN $names RETURN a.name AS a, collect(DISTINCT b.name) AS nbrs",
                names=seed_names,
            ):
                assoc_nbrs[r["a"]] = r["nbrs"]
            for r in s.run(
                f"MATCH (e:{config.EPISODE_LABEL})-->(x:{config.ENTITY_LABEL}) "
                "WHERE e.id IN $ids RETURN e.id AS id, collect(DISTINCT x.name) AS ents",
                ids=related_ids,
            ):
                ep_ents[r["id"]] = r["ents"]
    store.close()

    # --- グラフ構築 -----------------------------------------------------
    net = Network(height="820px", width="100%", directed=True, notebook=False)
    net.set_options(_OPTIONS)

    def ent_id(name: str) -> str:
        return f"ent:{name}"

    # ① クエリ
    net.add_node("QUERY", label=query, level=0, color=C_QUERY, shape="box")

    if seed_ep is None:
        net.add_node("RESPONSE", label=response[:60], title=response, level=1,
                     color=C_RESP, shape="box")
        net.add_edge("QUERY", "RESPONSE", label="想起失敗→正直に応答")
    else:
        # ② seed エピソード
        net.add_node(seed_ep.id, label=seed_ep.event[:22], title=seed_ep.as_text(),
                     level=1, color=C_EP, shape="ellipse")
        net.add_edge("QUERY", seed_ep.id, label=f"① ACT-R想起 A(m)={seed.total_a:.1f}")

        # ③ seed のエンティティ
        seed_set = set(seed_names)
        for e in seed_ep.entities:
            net.add_node(ent_id(e.name), label=e.name, title=f"{e.name} ({e.type})",
                         level=2, color=C_ENT, shape="dot")
            net.add_edge(seed_ep.id, ent_id(e.name), label=e.relation)

        # ④ ASSOC 連想エンティティ（関連エピソードに繋がるものだけ）＋ ⑤ 関連エピソード
        related_ent_union = set().union(*[set(ep_ents.get(i, [])) for i in related_ids]) if related_ids else set()
        added_aent: set[str] = set()
        for ep in related_eps:
            net.add_node(ep.id, label=ep.event[:22], title=ep.as_text(),
                         level=4, color=C_EP, shape="ellipse")
            ents_here = set(ep_ents.get(ep.id, []))
            # 直接共有（seedエンティティ経由）
            for nm in ents_here & seed_set:
                net.add_edge(ent_id(nm), ep.id, label="言及", color="#9aa0a6")
            # ASSOC 連想経由
            for s_name, nbrs in assoc_nbrs.items():
                for nb in nbrs:
                    if nb in ents_here and nb not in seed_set:
                        if nb not in added_aent:
                            net.add_node(ent_id(nb), label=nb, title=f"{nb}（連想）",
                                         level=3, color=C_AENT, shape="dot")
                            added_aent.add(nb)
                        net.add_edge(ent_id(s_name), ent_id(nb), label="ASSOC連想", color="#f032e6")
                        net.add_edge(ent_id(nb), ep.id, label="言及", color="#9aa0a6")

        # ⑥ 応答
        net.add_node("RESPONSE", label=response[:70] + "…", title=response,
                     level=5, color=C_RESP, shape="box")
        net.add_edge(seed_ep.id, "RESPONSE", label="応答に使用", color="#3cb44b")
        for ep in related_eps:
            net.add_edge(ep.id, "RESPONSE", color="#3cb44b")

    net.write_html(output_path, notebook=False)
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html.replace("</body>", _LEGEND + "</body>", 1))

    print(f"Flow written to {output_path}")
    print(f"  クエリ: {query}")
    if seed_ep:
        print(f"  seed: {seed_ep.id} ({seed_ep.event[:30]})")
        print(f"  関連: {', '.join(related_ids)}")
    print(f"  応答: {response[:60]}…")
    return output_path


if __name__ == "__main__":
    config.load_env()
    q = sys.argv[1] if len(sys.argv) > 1 else "電車に乗るのは平気ですか？"
    out = os.path.join(os.path.dirname(__file__), "flow.html")
    export_flow(q, output_path=out)
