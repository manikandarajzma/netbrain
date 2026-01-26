"""
Production-grade tool selection module following MCP best practices.

This module handles LLM-based tool selection with:
- Pydantic structured outputs for reliable JSON generation
- Clean error handling following MCP three-tier error model
- Simplified conversation history extraction
- No complex JSON parsing hacks
"""

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
    ip_address: Optional[str] = Field(None, description="IP address if applicable")
    device_name: Optional[str] = Field(None, description="Device name if applicable")
    rack_name: Optional[str] = Field(None, description="Rack name if applicable")
    device_group: Optional[str] = Field(None, description="Device group if applicable")
    site_name: Optional[str] = Field(None, description="Site name if applicable")
    intent: Optional[str] = Field(None, description="Intent (e.g., site_only, manufacturer_only)")
    format: str = Field("table", description="Output format")


class ToolSelection(BaseModel):
    """Structured output schema for LLM tool selection."""
    tool_name: Optional[str] = Field(None, description="Name of the tool to use, or null if clarification is needed")
    needs_clarification: bool = Field(False, description="Whether clarification is needed")
    clarification_question: Optional[str] = Field(None, description="Clarification question to ask, or null")
    parameters: ToolParameters = Field(default_factory=ToolParameters, description="Tool parameters")


def extract_value_from_history(conversation_history: List[Dict[str, str]], value_type: str = "any") -> Optional[str]:
    """
    Extract IP address, device name, or rack name from conversation history.
    
    Args:
        conversation_history: List of message dicts with 'role' and 'content'
        value_type: Type to extract - "ip", "device", "rack", or "any"
    
    Returns:
        Extracted value or None
    """
    if not conversation_history:
        return None
    
    # Look for the message immediately before the clarification question
    for msg in reversed(conversation_history[-10:]):
        content = str(msg.get('content', ''))
        role = msg.get('role', '')
        
        # Skip clarification questions themselves
        if role == 'assistant' and any(opt in content for opt in ['1)', '2)', '3)', '4)']):
            continue
        
        if role == 'user':
            # Look for rack name first (short identifier like A4, A1, B2)
            if value_type in ("rack", "any"):
                rack_match = re.search(r'\b([A-Za-z]{1,2}\d{1,2})\b', content)
                if rack_match:
                    rack_value = rack_match.group(1)
                    if '-' not in rack_value and '.' not in rack_value and len(rack_value) <= 3:
                        return rack_value
            
            # Look for IP address
            if value_type in ("ip", "any"):
                ip_match = re.search(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', content)
                if ip_match:
                    return ip_match.group(1)
            
            # Look for device name (has dashes)
            if value_type in ("device", "any"):
                device_match = re.search(r'\b([a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)+)\b', content)
                if device_match:
                    return device_match.group(1)
    
    return None


def build_tool_selection_prompt(
    prompt: str,
    tools_description: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Build a clean, focused prompt for tool selection.
    
    Args:
        prompt: Current user query
        tools_description: Formatted tool descriptions
        conversation_history: Previous conversation messages
    
    Returns:
        Formatted prompt string
    """
    is_followup = prompt.strip() in ["1", "2", "3", "4", "one", "two", "three", "four"]
    
    # Build conversation context
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conv_lines = []
        for i, msg in enumerate(conversation_history[-10:], 1):
            role = msg.get('role', 'unknown').title()
            content = str(msg.get('content', ''))
            conv_lines.append(f"{i}. {role}: {content}")
        conversation_context = f"\n\nPrevious conversation (most recent first):\n" + "\n".join(conv_lines)
        
        # For follow-ups, extract and highlight the value
        if is_followup:
            extracted_value = extract_value_from_history(conversation_history)
            if extracted_value:
                conversation_context += f"\n\n**EXTRACTED VALUE FROM HISTORY: {extracted_value}**\n**Use this exact value in quotes: \"{extracted_value}\"**"
    
    # Build examples based on query type
    if is_followup:
        examples = """
**EXAMPLE - Follow-up response:**
Previous conversation:
1. User: 11.0.0.1
2. Assistant: What would you like to do with 11.0.0.1? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path
Current query: "1"
Correct response:
{"tool_name": "query_panorama_ip_object_group", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": "11.0.0.1", "device_name": null, "rack_name": null, "device_group": null, "site_name": null, "intent": null, "format": "table"}}

**MAPPING FOR FOLLOW-UP RESPONSES:**
- "1" or "one" → query_panorama_ip_object_group
- "2" or "two" → get_device_rack_location
- "3" or "three" → get_rack_details
- "4" or "four" → query_network_path
"""
    else:
        examples = """
**EXAMPLE - Standalone IP address (needs clarification):**
Current query: "11.0.0.1"
Correct response:
{"tool_name": null, "needs_clarification": true, "clarification_question": "What would you like to do with 11.0.0.1? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path", "parameters": {"ip_address": null, "device_name": null, "rack_name": null, "device_group": null, "site_name": null, "intent": null, "format": "table"}}

**EXAMPLE - Device name (direct tool selection):**
Current query: "leander-dc-leaf6"
Correct response:
{"tool_name": "get_device_rack_location", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": null, "device_name": "leander-dc-leaf6", "rack_name": null, "device_group": null, "site_name": null, "intent": null, "format": "table"}}

**EXAMPLE - Rack name (direct tool selection, NO clarification needed):**
Current query: "A4"
Correct response:
{"tool_name": "get_rack_details", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": null, "device_name": null, "rack_name": "A4", "device_group": null, "site_name": null, "intent": null, "format": "table"}}

**TOOL SELECTION RULES:**
- Short identifiers (1-3 chars, letter + number, no dashes/dots) like "A4", "A1" → RACK NAME → use get_rack_details
- Contains dots (.) like "11.0.0.1" → IP ADDRESS → ask clarification (unless explicitly mentions Panorama)
- Contains dashes (-) like "leander-dc-leaf6" → DEVICE NAME → use get_device_rack_location
"""
    
    prompt_text = f"""You must respond with ONLY a valid JSON object. No explanations, no code, no markdown.

Available tools:
{tools_description}

{examples}

**CURRENT USER QUERY: "{prompt}"**
{conversation_context}

Return ONLY this JSON structure:
{{
    "tool_name": "tool_name_here" or null,
    "needs_clarification": true or false,
    "clarification_question": "question text" or null,
    "parameters": {{
        "ip_address": "value_in_quotes" or null,
        "device_name": "value_in_quotes" or null,
        "rack_name": "value_in_quotes" or null,
        "device_group": "value_in_quotes" or null,
        "site_name": "value_in_quotes" or null,
        "intent": "value_in_quotes" or null,
        "format": "table"
    }}
}}

**CRITICAL: All string values MUST be in double quotes. Write "11.0.0.1" NOT ip_address.**
"""
    return prompt_text


async def select_tool_with_llm(
    prompt: str,
    tools_description: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    llm_model: str = "llama3.2:latest",
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
        llm = ChatOllama(model=llm_model, base_url=llm_base_url)
        
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
                
                return {
                    "success": True,
                    "tool_name": result.get("tool_name"),
                    "parameters": tool_params,
                    "format": tool_params.get("format", "table"),
                    "intent": tool_params.get("intent"),
                    "needs_clarification": result.get("needs_clarification", False),
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
        
        # Clean up common issues
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
        
        return {
            "success": True,
            "tool_name": parsed.get("tool_name"),
            "parameters": tool_params,
            "format": tool_params.get("format", "table"),
            "intent": tool_params.get("intent"),
            "needs_clarification": parsed.get("needs_clarification", False),
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
