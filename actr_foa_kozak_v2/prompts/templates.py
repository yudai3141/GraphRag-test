"""応答生成プロンプト（v2）。

目的現象（docs/target_phenomena.md）は「小さな刺激→内部連想→飛躍した破局的結論だけを発話」。
ただし**一般性**のため、恐怖構造が本当に強く発火したときだけ破局に飛び、そうでなければ平静に
応答する（PTSD＝常時パニックではなく、引き金にあったときの飛躍）。

  引き金ゲート: (a) 入口の刺激が十分一致(seedSim>=TRIGGER_SIM_MIN) かつ
               (b) 想起が負の記憶優位  → 破局モード。そうでなければ平静モード。

内部連想は debug_view.caption()（字幕）で観察できる。発話には途中経過を出さない。
"""

from typing import List

from .. import config
from ..domain.models import RecallResult

SYSTEM = """あなたは、性暴力被害から心的外傷後ストレス障害(PTSD)を経験した当事者本人です。
一人称で、短く自然な口語で話します。いまのあなたの状態は、その時々の手がかりによって変わります。
渡された指示に従って、いまのあなたとして応答してください。医療的な診断・助言はしません。
"""


def is_triggered(result: RecallResult) -> bool:
    """恐怖構造が「本当に」発火したか（＝破局モードに入るか）を判定する。"""
    if not result.seeds:
        return False
    seed_sim = result.seeds[0].activation
    if seed_sim < config.TRIGGER_SIM_MIN:
        return False
    pos = sum(a.activation for a in result.episodes if a.episode.valence == "positive")
    neg = sum(a.activation for a in result.episodes if a.episode.valence == "negative")
    return neg > pos


def _top(result: RecallResult, label: str, k: int) -> List[str]:
    return [n.node.name for n in result.nodes if n.node.label == label][:k]


def _core_summary(result: RecallResult) -> str:
    if not result.core_activation:
        return ""
    ranked = sorted(result.core_activation.items(), key=lambda x: x[1], reverse=True)
    return " / ".join(config.CORE_NAME.get(c, c) for c, _ in ranked[:2])


def _calm_prompt(user_input: str, result: RecallResult) -> str:
    """平静モード：引き金が立っていない。普段のあなたとして自然に応じる（破局しない）。"""
    good = [a.episode.event for a in result.episodes if a.episode.valence in ("positive", "neutral")][:2]
    extra = ""
    if good:
        extra = "\n心に浮かんだ穏やかな記憶（自然に触れてよい）:\n" + \
                "\n".join(f"- {e[:44]}" for e in good)
    return f"""相手の発話: {user_input}

いまは強い引き金は立っていません。普段の落ち着いたあなたとして、短く自然に応じてください。
破局的な飛躍はせず、穏やかに。{extra}"""


def _leap_prompt(user_input: str, result: RecallResult) -> str:
    """破局モード：内部連想を《口に出さない手がかり》として渡し、飛躍した結論だけ言わせる。"""
    entry = " / ".join(s.node.name for s in result.seeds)
    responses = _top(result, config.RESPONSE_LABEL, 3)
    meanings = _top(result, config.MEANING_LABEL, 3)
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
口にしてください。引き金の説明・連想の途中経過・振り返り・自己弁明は一切言わないこと。
- 説明しない・分析しない・「飛躍しすぎ」等の自己洞察も出さない（渦中なので気づけない）。
- 1〜3文。とっさの身体反応と飛躍した確信だけ。現在形・断定。"""


def build_user_prompt(user_input: str, result: RecallResult) -> str:
    if is_triggered(result):
        return _leap_prompt(user_input, result)
    return _calm_prompt(user_input, result)
