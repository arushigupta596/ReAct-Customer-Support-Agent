"""A scripted chat model so graph tests run without any API calls."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted AIMessages in order and records what it was sent."""

    script: list[AIMessage]
    calls: list[dict] = []  # noqa: RUF012  pydantic copies field defaults per instance

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        self.calls.append({"tool_choice": tool_choice, "n_tools": len(tools)})
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any):
        if not self.script:
            tail = "\n".join(f"  {m.type}: {str(m.content)[:200]}" for m in messages[-4:])
            raise AssertionError(
                f"ScriptedChatModel ran out of scripted responses. Last messages:\n{tail}"
            )
        self.calls[-1]["messages"] = list(messages)
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])
