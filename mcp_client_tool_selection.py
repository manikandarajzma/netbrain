"""
Production-grade tool selection module following MCP best practices.

This module handles LLM-based tool selection with:
- Pydantic structured outputs for reliable JSON generation
- Clean error handling following MCP three-tier error model
- Simplified conversation history extraction
- No complex JSON parsing hacks
"""

import asyncio
import json
import re
import sys
from typing import Optional, Dict, Any, List
from langchain_ollama import ChatOllama

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    Field = None


class ToolParameters(BaseModel):
    """Tool parameters schema."""
    ip_address: Optional[str] = Field(
        None,
        description="IP or CIDR from the user query (REQUIRED when tool is query_panorama_ip_object_group or get_splunk_recent_denies; e.g. '11.0.0.0/24', '10.0.0.1')",
    )
    device_name: Optional[str] = Field(None, description="Device name if applicable")
    rack_name: Optional[str] = Field(None, description="Rack name if applicable")
    expected_rack: Optional[str] = Field(None, description="Expected rack name when user asks 'is device X in rack Y?' (e.g., 'A1', 'B4')")
    address_group_name: Optional[str] = Field(None, description="Address group name if applicable (e.g., 'leander_web', 'web_servers')")
    device_group: Optional[str] = Field(None, description="Device group if applicable")
    site_name: Optional[str] = Field(None, description="Site name if applicable")
    intent: Optional[str] = Field(None, description="Intent (e.g., site_only, manufacturer_only)")
    source: Optional[str] = Field(None, description="Source IP address for network path queries")
    destination: Optional[str] = Field(None, description="Destination IP address for network path queries")
    protocol: Optional[str] = Field(None, description="Protocol for network path queries (e.g., 'TCP', 'UDP')")
    port: Optional[str] = Field(None, description="Port number for network path queries")
    limit: Optional[int] = Field(None, description="Maximum number of results to return (extract from 'latest N', 'recent N', 'last N events', etc.; e.g., 'latest 10' → 10, 'recent 5 events' → 5)")
    format: str = Field("table", description="Output format")


class ToolSelection(BaseModel):
    """Structured output schema for LLM tool selection."""
    entity_analysis: Optional[str] = Field(None, description="Analysis of what entity type is in the query (e.g., 'has dashes → device name')")
    tool_name: Optional[str] = Field(None, description="Name of the tool to use, or null if clarification is needed")
    needs_clarification: bool = Field(False, description="Whether clarification is needed")
    clarification_question: Optional[str] = Field(None, description="Clarification question to ask, or null")
    parameters: ToolParameters = Field(default_factory=ToolParameters, description="Tool parameters")


def extract_value_from_history(conversation_history: List[Dict[str, str]], value_type: str = "any") -> Optional[str]:
    """
    Extract values from conversation history for follow-up responses.
    
    This function is kept for backward compatibility but no longer uses pattern matching.
    The LLM should extract values directly from the conversation history in the prompt.
    
    Args:
        conversation_history: List of message dicts with 'role' and 'content'
        value_type: Type to extract - "ip", "device", "rack", or "any"
    
    Returns:
        None - LLM should extract values from conversation history in the prompt
    """
    # No pattern matching - rely entirely on LLM to extract from conversation history
    return None


def build_tool_selection_prompt(
    prompt: str,
    tools_description: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Build prompt for tool selection using only tool descriptions (no rules or examples).
    """
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conv_lines = []
        for i, msg in enumerate(conversation_history[-10:], 1):
            role = msg.get('role', 'unknown').title()
            content = str(msg.get('content', ''))
            conv_lines.append(f"{i}. {role}: {content}")
        conversation_context = "\n\nPrevious conversation (most recent first):\n" + "\n".join(conv_lines)

    prompt_text = f"""You are a tool selection expert for network infrastructure queries. Your job is to select the MOST APPROPRIATE tool based on the user's query.

DECISION RULES (apply in order):

1. IDENTIFY THE ENTITY TYPE IN THE QUERY:
   - Contains DOTS (.)? → It's an IP address (e.g., "11.0.0.1", "192.168.1.1")
   - Contains DASHES (-) and is long (15+ chars)? → It's a DEVICE NAME (e.g., "leander-dc-border-leaf1", "roundrock-dc-leaf1")
   - Short alphanumeric (1-4 chars, no dots/dashes)? → It's a RACK NAME (e.g., "A4", "B12")
   - Multiple words without dots/dashes? → Might be a SITE NAME (e.g., "Leander", "Round Rock DC")

2. MATCH ENTITY TYPE TO TOOL:
   - DEVICE NAME (has dashes) + "where/rack/location" → get_device_rack_location
   - DEVICE NAME + "is in rack X?" (yes/no question) → get_device_rack_location, extract expected_rack from the question
   - "LIST" + "racks" (plural) + site → list_racks (shows ALL racks at a site)
   - RACK NAME (short, no dashes) + "rack/details/utilization" → get_rack_details (even if multiple sites mentioned)
   - RACK NAME + multiple sites (e.g., "rack A4 in leander round rock") → get_rack_details with rack_name="A4", site_name=null (server will ask which site)
   - IP ADDRESS + "address group/panorama/object" → query_panorama_ip_object_group
   - TWO IP ADDRESSES + "path/traffic/allowed" → check_path_allowed or query_network_path

CRITICAL DISTINCTION:
   - "List racks at Leander" → list_racks (PLURAL - shows ALL racks)
   - "Rack A4" or "Rack A4 at Leander" → get_rack_details (SINGULAR - shows ONE rack)

3. NEVER ASK FOR CLARIFICATION IF:
   - The query clearly matches a pattern above
   - You can extract all required parameters
   - The entity type is unambiguous (e.g., has dashes = device name, has dots = IP)

4. ONLY ASK FOR CLARIFICATION IF:
   - The query is genuinely ambiguous
   - Required parameters are missing AND cannot be inferred

EXAMPLES:
- "where is leander-dc-border-leaf1 racked?" → DEVICE NAME (has dashes) → get_device_rack_location, device_name="leander-dc-border-leaf1", expected_rack=null
- "leander-dc-border-leaf1 is in A1?" → DEVICE NAME (has dashes) + yes/no question → get_device_rack_location, device_name="leander-dc-border-leaf1", expected_rack="A1" (extract the rack name "A1" from the question)
- "is roundrock-dc-leaf1 in rack B4?" → DEVICE NAME + yes/no question → get_device_rack_location, device_name="roundrock-dc-leaf1", expected_rack="B4" (extract the rack name "B4" from the question)
- "rack A4" → SINGULAR "rack" + specific name → get_rack_details, rack_name="A4", site_name=null
- "rack A4 in leander round rock" → SINGULAR "rack" + specific name + multiple sites → get_rack_details, rack_name="A4", site_name=null (server will ask which site)
- "rack A4 in Leander" → SINGULAR "rack" + specific name + single site → get_rack_details, rack_name="A4", site_name="Leander"
- "list racks at Leander" → PLURAL "racks" + NO specific rack name → list_racks, site_name="Leander" (shows ALL racks at Leander)
- "what racks are in Leander" → PLURAL "racks" + NO specific rack name → list_racks, site_name="Leander" (shows ALL racks)
- "show all racks at Round Rock" → PLURAL "racks" + NO specific rack name → list_racks, site_name="Round Rock"
- "list all racks" → PLURAL "racks" + NO site → list_racks, site_name=null (shows ALL racks everywhere)
- "what address group is 11.0.0.1 part of?" → IP ADDRESS (has dots) → query_panorama_ip_object_group, ip_address="11.0.0.1"
- "latest 10 events for 10.0.0.250" → IP ADDRESS + "latest N" → get_splunk_recent_denies, ip_address="10.0.0.250", limit=10
- "recent 5 deny events for 192.168.1.1" → IP ADDRESS + "recent N" → get_splunk_recent_denies, ip_address="192.168.1.1", limit=5
- "last 20 denies for 10.0.0.1" → IP ADDRESS + "last N" → get_splunk_recent_denies, ip_address="10.0.0.1", limit=20
- "denies for 10.0.0.250" → IP ADDRESS + no number → get_splunk_recent_denies, ip_address="10.0.0.250", limit=null (uses default)

CRITICAL for yes/no questions like "is X in rack Y?":
- Extract the rack name (e.g., "A1", "B4") into expected_rack parameter
- Do NOT put "is in rack X?" in the intent field
- Example: "leander-dc-border-leaf1 is in A1" → expected_rack="A1" (NOT intent="is in A1")

CRITICAL for queries with numeric limits (latest/recent/last N):
- Extract the number from "latest N", "recent N", "last N events", etc. into limit parameter
- Examples: "latest 10" → limit=10, "recent 5 events" → limit=5, "last 20 denies" → limit=20
- If no number specified, leave limit=null to use the default

Current user query: "{prompt}"
{conversation_context}

**AVAILABLE TOOLS (numbered list):**
{tools_description}

**YOUR TASK:**
1. Identify the entity type in the query (IP vs device name vs rack name vs site)
2. Match it to the appropriate tool from the list above
3. Extract parameters from the query
   - For yes/no questions like "is X in rack Y?": extract the rack name (e.g., "A1") into expected_rack
   - For queries with "latest N", "recent N", "last N": extract the number into limit
   - Do NOT use the intent field for yes/no questions

RESPOND WITH ONLY A VALID JSON OBJECT (no markdown, no explanation).
Format: First think through the entity type, then respond.

{{
    "entity_analysis": "What entity is in the query? (e.g., 'leander-dc-border-leaf1' has dashes and is long → DEVICE NAME)",
    "tool_name": "<exact tool name from the list above>" or null,
    "needs_clarification": true or false,
    "clarification_question": "question text" or null,
    "parameters": {{
        "ip_address": null or "value",
        "device_name": null or "value",
        "rack_name": null or "value",
        "expected_rack": null or "A1" or "B4" (ONLY the rack name when user asks 'is X in rack Y?', NOT a description),
        "address_group_name": null or "value",
        "device_group": null or "value",
        "site_name": null or "value",
        "intent": null (DO NOT use for yes/no questions),
        "source": null or "value",
        "destination": null or "value",
        "protocol": null or "value",
        "port": null or "value",
        "limit": null or 10 or 20 (extract number from "latest N", "recent N", "last N events"),
        "format": "table"
    }}
}}
"""
    return prompt_text



async def select_tool_with_llm(
    prompt: str,
    tools_description: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    llm_model: str = "qwen2.5:14b",  # Better reasoning than llama3.1:8b
    llm_base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Use LLM to select appropriate tool based on user query.
    
    Follows MCP best practices:
    - Uses Pydantic structured outputs for reliable JSON
    - Clean error handling
    - No complex JSON parsing hacks
    
    Args:
        prompt: User query
        tools_description: Formatted tool descriptions
        conversation_history: Previous conversation messages
        llm_model: LLM model name
        llm_base_url: LLM base URL
    
    Returns:
        Dict with keys: success, tool_name, parameters, format, intent, needs_clarification, clarification_question, error
    """
    try:
        # Check if Ollama is accessible before creating LLM instance
        import socket
        from urllib.parse import urlparse
        parsed_url = urlparse(llm_base_url)
        host = parsed_url.hostname or 'localhost'
        port = parsed_url.port or 11434
        
        # Try to connect to Ollama server
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # 2 second timeout
            result = sock.connect_ex((host, port))
            sock.close()
            if result != 0:
                return {
                    "success": False,
                    "error": f"Ollama server is not running or not accessible at {llm_base_url}. Please start Ollama with 'ollama serve' or check if it's running on a different port.",
                    "tool_name": None,
                    "parameters": {},
                    "needs_clarification": None
                }
        except Exception as conn_error:
            return {
                "success": False,
                "error": f"Cannot connect to Ollama server at {llm_base_url}: {str(conn_error)}. Please ensure Ollama is running with 'ollama serve'.",
                "tool_name": None,
                "parameters": {},
                "needs_clarification": None
            }
        
        llm = ChatOllama(model=llm_model, base_url=llm_base_url, temperature=0.0)
        
        # Use Pydantic structured outputs if available
        if PYDANTIC_AVAILABLE and BaseModel:
            try:
                structured_llm = llm.with_structured_output(ToolSelection)
                prompt_text = build_tool_selection_prompt(prompt, tools_description, conversation_history)
                
                print(f"DEBUG: Using Pydantic structured outputs", file=sys.stderr, flush=True)
                response = structured_llm.invoke(prompt_text)
                
                # Convert Pydantic model to dict
                result = response.model_dump()
                
                # Extract parameters
                params = result.get("parameters", {})
                if isinstance(params, dict):
                    tool_params = params
                else:
                    tool_params = params.model_dump() if hasattr(params, 'model_dump') else {}
                
                tool_name = result.get("tool_name")
                needs_clarification = result.get("needs_clarification", False)
                entity_analysis = result.get("entity_analysis", "")
                print(f"DEBUG: LLM Analysis: {entity_analysis}", file=sys.stderr, flush=True)
                print(f"DEBUG: Pydantic result - tool_name: {tool_name}, needs_clarification: {needs_clarification}, clarification_question: {result.get('clarification_question')}", file=sys.stderr, flush=True)

                return {
                    "success": True,
                    "tool_name": tool_name,
                    "parameters": tool_params,
                    "format": tool_params.get("format", "table"),
                    "intent": tool_params.get("intent"),
                    "needs_clarification": needs_clarification,
                    "clarification_question": result.get("clarification_question")
                }
            except Exception as pydantic_error:
                print(f"DEBUG: Pydantic structured output failed: {str(pydantic_error)}", file=sys.stderr, flush=True)
                print(f"DEBUG: Falling back to manual JSON parsing", file=sys.stderr, flush=True)
                # Fall through to manual parsing
        
        # Fallback: Manual JSON parsing (simplified, no complex hacks)
        prompt_text = build_tool_selection_prompt(prompt, tools_description, conversation_history)
        response = llm.invoke(prompt_text)
        content = response.content if hasattr(response, 'content') else str(response)
        
        print(f"DEBUG: LLM response (first 500 chars): {content[:500]}", file=sys.stderr, flush=True)
        
        # Simple JSON extraction - find first { to last }
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            return {
                "success": False,
                "error": f"No valid JSON found in LLM response. Response: {content[:200]}..."
            }
        
        json_str = content[first_brace:last_brace + 1]
        
        # Clean up Python boolean/None values to JSON (minimal cleanup only)
        json_str = re.sub(r'\bNone\b', 'null', json_str)
        json_str = re.sub(r'\bTrue\b', 'true', json_str)
        json_str = re.sub(r'\bFalse\b', 'false', json_str)
        
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse JSON: {str(e)}. JSON string: {json_str[:200]}..."
            }
        
        # Extract values
        tool_params = parsed.get("parameters", {})
        needs_clarification = parsed.get("needs_clarification", False)
        tool_name = parsed.get("tool_name")
        
        print(f"DEBUG: Parsed JSON - tool_name: {tool_name}, needs_clarification: {needs_clarification}, clarification_question: {parsed.get('clarification_question')}", file=sys.stderr, flush=True)
        
        return {
            "success": True,
            "tool_name": tool_name,
            "parameters": tool_params,
            "format": tool_params.get("format", "table"),
            "intent": tool_params.get("intent"),
            "needs_clarification": needs_clarification,
            "clarification_question": parsed.get("clarification_question")
        }
        
    except Exception as e:
        import traceback
        print(f"DEBUG: Error in tool selection: {str(e)}", file=sys.stderr, flush=True)
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {
            "success": False,
            "error": f"Error during tool selection: {str(e)}"
        }


async def synthesize_final_answer(
    user_prompt: str,
    tool_name: str,
    error_or_result: str | Dict[str, Any],
    *,
    llm_model: str = "qwen2.5:14b",  # Better reasoning than llama3.1:8b
    llm_base_url: str = "http://localhost:11434",
    timeout: float = 15.0,
) -> str:
    """
    Use the LLM to synthesize a short, user-friendly final answer after a failed tool call (or to summarize a raw error).
    Forces a coherent response instead of returning only the raw error message.
    """
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
3. Suggest one or two concrete things the user can try (e.g. check spelling, specify a site name, ensure the MCP server is running).

Reply with ONLY the final answer text, no prefix like "Answer:" or markdown."""
    try:
        llm = ChatOllama(model=llm_model, base_url=llm_base_url, temperature=0.3)
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
        print(f"DEBUG: synthesize_final_answer failed: {e}", file=sys.stderr, flush=True)
        return f"{err_msg}\n\n(You can retry or rephrase your question; if the problem continues, check mcp_server.log and backend connectivity.)"
