"""
Shared tool-calling loop for agents.
"""
import json
import logging

logger = logging.getLogger("atlas.agents.loop")


async def run_agent_loop(
    task: str,
    system_prompt: str,
    tools: list,
    max_iterations: int = 5,
) -> str:
    """
    Run a tool-calling loop.

    The LLM decides which tools to call and in what order.
    Iterates until the LLM stops issuing tool calls or max_iterations is reached.
    Returns the LLM's final natural-language response.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL

    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
    )
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=task)]
    last_response = None

    for i in range(max_iterations):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        last_response = response

        if not response.tool_calls:
            logger.info("Agent loop: finished after %d iteration(s)", i + 1)
            break

        for tc in response.tool_calls:
            name, args = tc["name"], tc["args"]
            logger.info("Agent loop: calling %s with %s", name, args)

            if name not in tool_map:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = await tool_map[name].ainvoke(args)
                except Exception as exc:
                    result = {"error": str(exc)}
                    logger.warning("Tool %s error: %s", name, exc)

            messages.append(ToolMessage(
                content=json.dumps(result) if not isinstance(result, str) else result,
                tool_call_id=tc["id"],
            ))

    if not last_response:
        return "No response generated."
    content = last_response.content
    # If the loop hit max_iterations while the LLM was still issuing tool calls,
    # last_response is a tool-call message with no text.  Make one final call
    # without tools bound so the LLM synthesises from the accumulated context.
    if not content and last_response.tool_calls:
        logger.warning("Agent loop hit max_iterations with pending tool calls — forcing synthesis")
        synthesis_response = await llm.ainvoke(messages)
        content = synthesis_response.content or "Investigation complete — no summary generated."
    return content or "Investigation complete — no summary generated."
