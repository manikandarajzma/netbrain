"""
MCP Tool Result Synthesis.

Provides synthesize_final_answer to turn tool errors into user-friendly text.
Tool selection is handled natively by the LLM via LangChain bind_tools() in chat_service.py.
"""

import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger("atlas.tool_selection")
from langchain_openai import ChatOpenAI

try:
    from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
except ImportError:
    from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL


async def synthesize_final_answer(
    user_prompt: str,
    tool_name: str,
    error_or_result: str | Dict[str, Any],
    *,
    llm_model: str = None,
    llm_base_url: str = None,
    timeout: float = 15.0,
) -> str:
    """
    Use the LLM to synthesize a short, user-friendly final answer after a failed tool call.
    """
    if llm_model is None:
        llm_model = OLLAMA_MODEL
    if llm_base_url is None:
        llm_base_url = OLLAMA_BASE_URL

    if isinstance(error_or_result, dict):
        err_msg = error_or_result.get("error") or error_or_result.get("message") or str(error_or_result)[:500]
    else:
        err_msg = str(error_or_result)[:500]

    prompt_text = f"""The user asked: "{user_prompt}"

The system tried to answer using the tool "{tool_name}" but got this result or error:
{err_msg}

Write a very short final answer (2-4 sentences) to show the user. Do the following:
1. Briefly state what was attempted.
2. Say why it failed or what went wrong in plain language.
3. Suggest one or two concrete things the user can try.

Reply with ONLY the final answer text, no prefix like "Answer:" or markdown."""

    try:
        llm = ChatOpenAI(model=llm_model, base_url=llm_base_url, temperature=0.3, api_key="docker")
        if hasattr(llm, "ainvoke"):
            response = await asyncio.wait_for(llm.ainvoke(prompt_text), timeout=timeout)
        else:
            response = await asyncio.wait_for(
                asyncio.to_thread(llm.invoke, prompt_text),
                timeout=timeout,
            )
        text = response.content if hasattr(response, "content") else str(response)
        return (text or err_msg).strip() or err_msg
    except asyncio.TimeoutError:
        return f"The query could not be completed in time. {err_msg}"
    except Exception as e:
        logger.error(f"synthesize_final_answer failed: {e}")
        return f"{err_msg}\n\n(You can retry or rephrase your question; if the problem continues, check mcp_server.log and backend connectivity.)"
