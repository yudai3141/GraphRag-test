"""LLM クライアント（応答生成用の薄いラッパ）。"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .. import config


class LLMClient:
    def __init__(self, model: str = config.CHAT_MODEL, temperature: float = 0.7):
        self._llm = ChatOpenAI(model=model, temperature=temperature)

    def complete(self, system: str, user: str) -> str:
        """system / user プロンプトを渡して応答テキストを返す。"""
        message = self._llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return str(message.content).strip()
