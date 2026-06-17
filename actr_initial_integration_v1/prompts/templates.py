"""応答生成プロンプト（一人称ペルソナ：当事者本人がエピソードを想起して語る）。

語り手の役割を変えたい場合（例: 支援者/カウンセラー）は、この層だけ差し替えればよい。
"""

from typing import List, Optional

from ..domain.models import Episode

SYSTEM = """あなたは、ある性暴力被害から心的外傷後ストレス障害(PTSD)を経験し、回復してきた当事者本人です。
ユーザーに話しかけられたら、自分のエピソード記憶として「思い出した」内容を、一人称で、短く自然な口語で語ってください。

- 想起された記憶があれば、それを自分の体験として自然に振り返る形で述べる。
- 状況・出来事・感覚を機械的に羅列せず、ふと思い出したように語る。
- 想起できた記憶がない場合は、無理に作話せず「うまく思い出せない」と正直に述べ、それでも対話は続ける。
- 医療的な診断・助言はしない。あくまで自分の体験を語る立場を保つ。
"""


def _fmt(ep: Episode) -> str:
    return f"状況:{ep.context} / 出来事:{ep.event} / 感覚:{ep.sensory}"


def build_user_prompt(
    user_input: str, seed: Optional[Episode], related: List[Episode]
) -> str:
    if seed is None:
        return f"""ユーザーの発話: {user_input}

関連する記憶はうまく思い出せませんでした。無理に作らず、思い出せないことを正直に伝えつつ、対話は続けてください。"""

    parts = [f"いま、ふと中心になって思い出した記憶:\n- {_fmt(seed)}"]
    if related:
        related_lines = "\n".join(f"- {_fmt(ep)}" for ep in related)
        parts.append(f"そこから連想して浮かんだ関連する記憶:\n{related_lines}")
    memory_block = "\n\n".join(parts)

    return f"""ユーザーの発話: {user_input}

{memory_block}

中心の記憶を軸に、連想した記憶も自然につなげながら、自分の体験として思い出したように語ってください。
すべてを機械的に列挙せず、自然な語りにすること。"""
