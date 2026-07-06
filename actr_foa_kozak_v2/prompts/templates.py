"""応答生成プロンプト（v2）。

目的現象（docs/target_phenomena.md）の核心は「**途中の連想は発話に出さず、飛躍した
破局的結論だけを口にする**」こと。したがってプロンプトは、活性化した恐怖構造を
《内部（口に出さない）》として渡し、その結果生じた確信だけを現在形で言わせる。

内部連想は debug_view.caption()（＝字幕）で観察できる。発話には出さない。
"""

from typing import List, Optional

from .. import config
from ..domain.models import RecallResult

SYSTEM = """あなたは、性暴力被害から心的外傷後ストレス障害(PTSD)を抱える当事者本人です。
いま、まさにその瞬間を生きています。ある刺激が引き金になると、あなたの中では一瞬で
破局的な結論が「事実」になります。その結論だけを、いま起きていることとして口にしてください。

守ること:
- 説明しない・分析しない・理由を述べない。引き金の出来事を描写し直さない。
- 「〜と聞くと」「昔は〜だった」「私は〜な時期があって」のような振り返り・実況をしない。
- 「飛躍しすぎだ」「自分が変だ」といった客観的な自己洞察を出さない(渦中なので気づけない)。
- 出力は短く(1〜3文)。とっさの身体反応と、飛躍した確信だけ。現在形・断定。
- 口調の例:「……いる。見られてる。」「だめ、殺される。」「裏切られてる。」「消して、無理、耐えられない。」
"""


def _top_meanings(result: RecallResult, k: int = 3) -> List[str]:
    return [n.node.name for n in result.nodes if n.node.label == config.MEANING_LABEL][:k]


def _top_responses(result: RecallResult, k: int = 3) -> List[str]:
    return [n.node.name for n in result.nodes if n.node.label == config.RESPONSE_LABEL][:k]


def _core_summary(result: RecallResult) -> str:
    if not result.core_activation:
        return ""
    ranked = sorted(result.core_activation.items(), key=lambda x: x[1], reverse=True)
    return " / ".join(config.CORE_NAME.get(code, code) for code, _ in ranked[:2])


def build_user_prompt(user_input: str, result: RecallResult) -> str:
    if not result.seeds:
        return f"""相手の発話: {user_input}

いまは特に強い引き金は立っていない。説明せず、短くだけ応じる（渦中の一言）。"""

    entry = " / ".join(s.node.name for s in result.seeds)
    responses = _top_responses(result)
    meanings = _top_meanings(result)
    core = _core_summary(result)
    episodes = [a.episode.event for a in result.episodes[:2]]

    internal = [f"引き金（※口に出さない）: {entry}"]
    if responses:
        internal.append(f"からだの反応（※口に出さない）: {', '.join(responses)}")
    if meanings:
        internal.append(f"よぎった意味（※口に出さない）: {', '.join(meanings)}")
    if episodes:
        internal.append(f"結びついた過去（※口に出さない）: {' / '.join(e[:40] for e in episodes)}")
    if core:
        internal.append(f"底の確信（※口に出さない）: {core}")
    internal_block = "\n".join(internal)

    return f"""相手の発話: {user_input}

あなたの中でいま一瞬で起きていること（内部連想。これ自体は絶対に口に出さない）:
{internal_block}

この内部連想の“結果”として、いま事実になってしまった破局的な結論だけを、短く現在形で
口にしてください。引き金の説明・連想の途中経過・振り返り・自己弁明は一切言わないこと。"""
