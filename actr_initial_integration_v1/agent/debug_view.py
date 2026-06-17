"""内部処理を人間に分かりやすく表示する DEBUG ビュー（二段想起）。

  ① ACT-R activation のランキング → seed 特定
  ② seed のエンティティ連想 → 関連エピソード取得
"""

from ..domain.models import RecallResult


def render(result: RecallResult, threshold: float, now: float, top_n: int = 8) -> None:
    seed = result.seed

    # --- ① ACT-R activation ---------------------------------------------
    print("\n" + "─" * 72)
    print(f"🧠 ① ACT-R activation で seed を特定   t={now:.0f}   閾値={threshold:+.2f}")
    print(f"   A(m) = B(m)[ベースレベル+時間減衰] + 類似度×重み + ノイズ")
    print(f"   {'':4}{'A(m)':>8} = {'B(m)':>7} + {'spread':>7} + {'noise':>6}   (sim)")
    for rank, c in enumerate(result.candidates[:top_n], 1):
        mark = "★seed" if (seed and c.episode.id == seed.episode.id) else (
            "  ◯ " if c.above_threshold else "  ✗ "
        )
        print(
            f"{mark}{rank:2d}.{c.total_a:8.3f} = {c.b_level:7.3f} + "
            f"{c.a_spreading:7.3f} + {c.noise:+6.3f}   (sim={c.similarity:.3f})"
        )
        print(f"        └ {c.episode.event[:46]}")

    if seed is None:
        print("   → 閾値を超える記憶なし（想起失敗）")
        print("─" * 72)
        return

    # --- ② 意味層の連想 -------------------------------------------------
    ents = ", ".join(f"{e.name}({e.type})" for e in seed.episode.entities) or "(なし)"
    print("─" * 72)
    print(f"🔗 ② seed={seed.episode.id} のエンティティから連想")
    print(f"   seed エンティティ: {ents}")
    if result.related:
        for a in result.related:
            print(f"   └ 共有{a.shared}  {a.episode.id}: {a.episode.event[:40]}")
    else:
        print("   └ 連想で引けた関連エピソードなし")
    ids = ", ".join([seed.episode.id] + [a.episode.id for a in result.related])
    print(f"   → 文脈に渡す記憶: {ids}")
    print("─" * 72)
