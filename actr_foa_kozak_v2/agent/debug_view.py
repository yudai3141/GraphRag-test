"""内部処理を人間に分かりやすく表示する DEBUG ビュー（拡散活性版）。

  ① クエリ→入口の刺激（Stimulus）
  ② 恐怖構造の拡散活性で活性化したノード（刺激/反応/意味）
  ③ 中核評価（危険/BAD/無力/終わらない）への収束
  ④ 想起された過去エピソード
"""

from .. import config
from ..domain.models import RecallResult

_ROLE_JP = {
    config.STIMULUS_LABEL: "刺激",
    config.RESPONSE_LABEL: "反応",
    config.MEANING_LABEL: "意味",
    config.CORE_LABEL: "中核",
}


def render(result: RecallResult) -> None:
    print("\n" + "─" * 72)
    print(f"🧠 恐怖構造の拡散活性   query=「{result.query}」")

    # ① 入口
    if not result.seeds:
        print("   → クエリに一致する刺激なし（想起の入口が立たず）")
        print("─" * 72)
        return
    entry = ", ".join(f"{s.node.name}({s.activation:.2f})" for s in result.seeds)
    print(f"   ① 入口の刺激: {entry}")

    # ② 活性ノード
    print("   ② 拡散で活性化したノード（活性降順・hop=到達ホップ）:")
    for n in result.nodes[:10]:
        role = _ROLE_JP.get(n.node.label, n.node.label)
        print(f"        {n.activation:6.3f}  [{role}] {n.node.name[:34]}  (hop{n.hop})")

    # ③ 中核評価への収束
    if result.core_activation:
        ranked = sorted(result.core_activation.items(), key=lambda x: x[1], reverse=True)
        bars = "  ".join(
            f"{config.CORE_NAME.get(c, c)}={a:.2f}" for c, a in ranked
        )
        print(f"   ③ 中核評価への収束: {bars}")

    # ④ 想起エピソード
    print("   ④ 想起した過去エピソード:")
    if result.episodes:
        for a in result.episodes:
            print(f"        {a.activation:6.3f}  {a.episode.id}[{a.episode.valence}]: "
                  f"{a.episode.event[:38]}  (hop{a.hop})")
    else:
        print("        （恐怖構造をたどって届いたエピソードなし）")
    print("─" * 72)
