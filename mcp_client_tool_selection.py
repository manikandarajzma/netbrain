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
    address_group_name: Optional[str] = Field(None, description="Address group name if applicable (e.g., 'leander_web', 'web_servers')")
    device_group: Optional[str] = Field(None, description="Device group if applicable")
    site_name: Optional[str] = Field(None, description="Site name if applicable")
    intent: Optional[str] = Field(None, description="Intent (e.g., site_only, manufacturer_only)")
    source: Optional[str] = Field(None, description="Source IP address for network path queries")
    destination: Optional[str] = Field(None, description="Destination IP address for network path queries")
    protocol: Optional[str] = Field(None, description="Protocol for network path queries (e.g., 'TCP', 'UDP')")
    port: Optional[str] = Field(None, description="Port number for network path queries")
    format: str = Field("table", description="Output format")


class ToolSelection(BaseModel):
    """Structured output schema for LLM tool selection."""
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
        
        # For follow-ups, instruct LLM to extract from conversation history
        if is_followup:
            conversation_context += "\n\n**IMPORTANT: For follow-up responses (like '1', '2', '3', '4'), extract the relevant value (IP address, device name, rack name, or address group name) from the conversation history above. Look at the user's message immediately before the clarification question.**"
        else:
            # For new queries, explicitly tell LLM to ignore old context
            conversation_context += "\n\n**CRITICAL: This is a NEW query, NOT a follow-up. Extract values ONLY from the CURRENT USER QUERY above. DO NOT use values from previous conversation history. Each new query should be processed independently.**"
    
    # Build examples based on query type
    if is_followup:
        examples = """
**EXAMPLE - Follow-up response:**
Previous conversation:
1. User: 11.0.0.1
2. Assistant: What would you like to do with 11.0.0.1? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path
Current query: "1"
Correct response:
{"tool_name": "query_panorama_ip_object_group", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": "11.0.0.1", "device_name": null, "rack_name": null, "address_group_name": null, "device_group": null, "site_name": null, "intent": null, "format": "table"}}

**MAPPING FOR FOLLOW-UP RESPONSES:**
- "1" or "one" → query_panorama_ip_object_group
- "2" or "two" → get_device_rack_location
- "3" or "three" → get_rack_details
- "4" or "four" → query_network_path (NOT query_panorama_network_path - the correct tool name is "query_network_path")
"""
    else:
        examples = """
**⚠️ DECISION TREE FOR ADDRESS GROUP QUERIES - READ THIS FIRST:**

Step 1: Does the query contain an IP address (has dots like "11.0.0.0/24", "11.0.0.1")?
  YES → Go to Step 2
  NO → Check if it contains an address group name (has underscores like "leander_web")

Step 2: What does the query ask?
  - "what address group is [IP] part of" → You have IP, want groups → use query_panorama_ip_object_group
  - "which group contains [IP]" → You have IP, want groups → use query_panorama_ip_object_group
  - "what object group is [IP] in" → You have IP, want groups → use query_panorama_ip_object_group

Step 3: Does the query contain an address group name (has underscores like "leander_web")?
  YES → What does the query ask?
    - "what IPs are in address group [NAME]" → You have group name, want IPs → use query_panorama_address_group_members
    - "list IPs in group [NAME]" → You have group name, want IPs → use query_panorama_address_group_members

**CRITICAL: The query structure tells you the direction:**
- IP → Groups = query_panorama_ip_object_group
- Group → IPs = query_panorama_address_group_members

**EXAMPLE - Standalone IP address (needs clarification):**
Current query: "11.0.0.1"
Correct response:
{"tool_name": null, "needs_clarification": true, "clarification_question": "What would you like to do with 11.0.0.1? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path", "parameters": {"ip_address": null, "device_name": null, "rack_name": null, "address_group_name": null, "device_group": null, "site_name": null, "intent": null, "format": "table"}}

**EXAMPLE - Device name (direct tool selection):**
Current query: "where is leander-dc-border-leaf1 racked"
Correct response:
{"tool_name": "get_device_rack_location", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": null, "device_name": "leander-dc-border-leaf1", "rack_name": null, "address_group_name": null, "device_group": null, "site_name": null, "intent": "rack_location_only", "format": "table"}}

**EXAMPLE - Device rack location query (variations):**
Current query: "What is the rack location of leander-dc-leaf6"
Correct response:
{"tool_name": "get_device_rack_location", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": null, "device_name": "leander-dc-leaf6", "rack_name": null, "address_group_name": null, "device_group": null, "site_name": null, "intent": "rack_location_only", "format": "table"}}

**CRITICAL RULE FOR DEVICE QUERIES:**
- If query contains a device name (has DASHES like "leander-dc-leaf6", "roundrock-dc-border-leaf1") AND asks about "rack location", "where is", "racked", or similar → use get_device_rack_location
- Device names have DASHES (-), NOT dots (.)
- "leander-dc-leaf6" has dashes → it's a device name → use get_device_rack_location
- "11.0.0.1" has dots → it's an IP address → do NOT use get_device_rack_location

**NOTE: Extract the EXACT device name from the query. If query says "leander-dc-leaf6", use "leander-dc-leaf6", NOT "leander-dc-border-leaf1" or any other device name.**

**EXAMPLE - Rack name (direct tool selection, NO clarification needed):**
Current query: "A4"
Correct response:
{"tool_name": "get_rack_details", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": null, "device_name": null, "rack_name": "A4", "address_group_name": null, "device_group": null, "site_name": null, "intent": null, "format": "table"}}

**EXAMPLE - Rack details query (direct tool selection, NO clarification needed):**
Current query: "rack details of A4"
Correct response:
{"tool_name": "get_rack_details", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": null, "device_name": null, "rack_name": "A4", "address_group_name": null, "device_group": null, "site_name": null, "intent": null, "format": "table"}}

**ABSOLUTE RULE: If the query contains a short identifier (1-3 characters, letter + number, no dashes, no dots) like "A4", "A1", "B2" AND the query mentions "rack", "rack details", "rack information", or similar rack-related terms, you MUST use get_rack_details with the rack name extracted from the query. Do NOT return tool_name as null.**
**ABSOLUTE RULE: If the query is just a short identifier (1-3 characters, letter + number, no dashes, no dots) like "A4", "A1", "B2" with no other context, you MUST use get_rack_details. Do NOT return tool_name as null.**

**⚠️ CRITICAL DISTINCTION - READ THIS FIRST FOR ADDRESS GROUP QUERIES:**

**DIRECTION MATTERS:**
- Query structure: "what address group is [IP] part of" or "which group contains [IP]" → You have an IP, want to find groups → use query_panorama_ip_object_group
- Query structure: "what IPs are in address group [NAME]" or "list IPs in group [NAME]" → You have a group name, want to list IPs → use query_panorama_address_group_members

**EXAMPLE - Find which address group an IP belongs to (IP → Groups):**
Current query: "what address group is 11.0.0.0/24 part of"
Step-by-step analysis:
1. Query contains IP address: "11.0.0.0/24" (has dots)
2. Query structure: "what address group is [IP] part of"
3. Direction: IP is INPUT, groups are OUTPUT
4. Decision: Use query_panorama_ip_object_group with ip_address="11.0.0.0/24"
Correct response:
{"tool_name": "query_panorama_ip_object_group", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": "11.0.0.0/24", "device_name": null, "rack_name": null, "address_group_name": null, "device_group": null, "site_name": null, "intent": null, "format": "table"}}

**WRONG EXAMPLE (DO NOT DO THIS):**
Current query: "what address group is 11.0.0.0/24 part of"
WRONG response: {"tool_name": "query_panorama_address_group_members", ...} ← This is WRONG because the query has an IP and asks which groups contain it, not which IPs are in a group

**EXAMPLE - List IPs in an address group (Group → IPs):**
Current query: "what other IPs are in the address group leander_web"
Step-by-step analysis:
1. Query contains address group name: "leander_web" (has underscore)
2. Query structure: "what IPs are in address group [NAME]"
3. Direction: Group name is INPUT, IPs are OUTPUT
4. Decision: Use query_panorama_address_group_members with address_group_name="leander_web"
Correct response:
{"tool_name": "query_panorama_address_group_members", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": null, "device_name": null, "rack_name": null, "address_group_name": "leander_web", "device_group": null, "site_name": null, "intent": null, "format": "table"}}

**EXAMPLE - Network path query (direct tool selection):**
Current query: "Find path from 10.0.0.1 to 10.0.1.1"
Step-by-step analysis:
1. Query contains "path from [IP] to [IP]" or "find path" or "network path"
2. Extract source IP: "10.0.0.1" (the IP after "from")
3. Extract destination IP: "10.0.1.1" (the IP after "to")
4. Decision: Use query_network_path with source="10.0.0.1" and destination="10.0.1.1"
Correct response:
{"tool_name": "query_network_path", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": null, "device_name": null, "rack_name": null, "address_group_name": null, "device_group": null, "site_name": null, "intent": null, "source": "10.0.0.1", "destination": "10.0.1.1", "protocol": null, "port": null, "format": "table"}}

**EXAMPLE - Network path query with protocol and port:**
Current query: "Find path from 10.0.0.1 to 10.0.1.1 using TCP port 80"
Step-by-step analysis:
1. Query contains "path from [IP] to [IP]"
2. Extract source IP: "10.0.0.1"
3. Extract destination IP: "10.0.1.1"
4. Extract protocol: "TCP" (mentioned in query)
5. Extract port: "80" (mentioned in query)
6. Decision: Use query_network_path with source="10.0.0.1", destination="10.0.1.1", protocol="TCP", port="80"
Correct response:
{"tool_name": "query_network_path", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": null, "device_name": null, "rack_name": null, "address_group_name": null, "device_group": null, "site_name": null, "intent": null, "source": "10.0.0.1", "destination": "10.0.1.1", "protocol": "TCP", "port": "80", "format": "table"}}

**EXAMPLE - Check if path is allowed/denied (policy check):**
Current query: "Is traffic from 10.0.0.1 to 10.0.1.1 on TCP port 80 allowed?"
Step-by-step analysis:
1. Query asks if traffic is "allowed" or "denied"
2. Extract source IP: "10.0.0.1"
3. Extract destination IP: "10.0.1.1"
4. Extract protocol: "TCP"
5. Extract port: "80"
6. Decision: Use check_path_allowed with source="10.0.0.1", destination="10.0.1.1", protocol="TCP", port="80"
Correct response:
{"tool_name": "check_path_allowed", "needs_clarification": false, "clarification_question": null, "parameters": {"ip_address": null, "device_name": null, "rack_name": null, "address_group_name": null, "device_group": null, "site_name": null, "intent": null, "source": "10.0.0.1", "destination": "10.0.1.1", "protocol": "TCP", "port": "80", "format": "table"}}

**CRITICAL DISTINCTION:**
- Query asks "is [traffic] allowed" or "is [traffic] denied" → use check_path_allowed (stops on policy denial)
- Query asks "find path" or "show path" → use query_network_path (continues even if denied)

**CRITICAL: For network path queries, use tool name "query_network_path" (NOT "query_panorama_network_path"). The correct tool name is "query_network_path".**

**ABSOLUTE RULE FOR ADDRESS GROUP QUERIES:**
- If query contains an IP address (has dots like "11.0.0.0/24", "11.0.0.1") AND asks "what address group is [IP] part of" or "which group contains [IP]" → use query_panorama_ip_object_group with ip_address parameter
- If query contains an address group name (has underscores like "leander_web") AND asks "what IPs are in address group [NAME]" → use query_panorama_address_group_members with address_group_name parameter
- DO NOT confuse these two - the direction of the query (IP→Groups vs Group→IPs) determines which tool to use

**TOOL SELECTION RULES (APPLY IN THIS ORDER):**
1. **FIRST: Check query direction for address group queries:**
   - Query asks "what address group is [IP] part of" or "which group contains [IP]" → IP is INPUT, groups are OUTPUT → use query_panorama_ip_object_group with ip_address
   - Query asks "what IPs are in address group [NAME]" or "list IPs in group [NAME]" → Group name is INPUT, IPs are OUTPUT → use query_panorama_address_group_members with address_group_name
   - **DO NOT confuse these - the query structure tells you the direction**

2. **Rack queries:**
   - Query mentions "rack", "rack details", "rack information", "rack utilization" AND contains short identifier (1-3 chars, letter + number, no dashes/dots) like "A4", "A1" → use get_rack_details
   - Short identifiers (1-3 chars, letter + number, no dashes/dots) like "A4", "A1" → use get_rack_details

3. **Device queries:**
   - Contains dashes (-) like "leander-dc-border-leaf1" → DEVICE NAME → use get_device_rack_location

4. **Standalone IP (no context):**
   - Contains dots (.) like "11.0.0.1" with no other context → ask clarification

5. **CRITICAL: Extract the EXACT value from the CURRENT USER QUERY. Do NOT use values from examples.**
"""
    
    # No pattern matching - rely entirely on LLM to extract values from the prompt
    # The tool descriptions and examples should be sufficient for the LLM to understand
    
    # Extract exact tool names from tools_description for validation
    import re
    tool_names = re.findall(r'Tool Name: ([^\n]+)', tools_description)
    available_tool_names = ', '.join(tool_names) if tool_names else 'See tool descriptions above'
    
    prompt_text = f"""You must respond with ONLY a valid JSON object. No explanations, no code, no markdown.

**⚠️ MANDATORY FIRST STEP - ANALYZE QUERY STRUCTURE:**
Before selecting any tool, analyze the CURRENT USER QUERY structure:
1. Does the query contain an IP address (has dots like "11.0.0.0/24", "11.0.0.1")?
2. Does the query contain an address group name (has underscores like "leander_web")?
3. What is the query asking for?
   - If query has IP and asks "what address group is [IP] part of" → IP is INPUT, groups are OUTPUT → use query_panorama_ip_object_group
   - If query has group name and asks "what IPs are in address group [NAME]" → Group is INPUT, IPs are OUTPUT → use query_panorama_address_group_members
   - **DO NOT confuse these two directions**

Available tools:
{tools_description}

**CRITICAL: Available tool names (use EXACTLY as shown): {available_tool_names}**
**You MUST use one of these exact tool names. Do NOT invent or modify tool names.**

{examples}

**CURRENT USER QUERY: "{prompt}"**
{conversation_context}

**CRITICAL - PRIORITIZE CURRENT QUERY:**
- **ALWAYS extract values from the CURRENT USER QUERY first. The current query is: "{prompt}"**
- **ONLY use conversation history if the current query is a follow-up response (like "1", "2", "3", "4")**
- **If the current query is a NEW question (not a follow-up), IGNORE previous conversation context and extract values ONLY from the current query**
- **DO NOT extract values from old queries in conversation history when processing a new, unrelated query**
- Use the tool descriptions to understand what each tool does and what parameters it needs. The examples are just for format reference.

Return ONLY this JSON structure:
{{
    "tool_name": "tool_name_here" or null,
    "needs_clarification": true or false,
    "clarification_question": "question text" or null,
    "parameters": {{
        "ip_address": "value_in_quotes" or null,
        "device_name": "value_in_quotes" or null,
        "rack_name": "value_in_quotes" or null,
        "address_group_name": "value_in_quotes" or null,
        "device_group": "value_in_quotes" or null,
        "site_name": "value_in_quotes" or null,
        "intent": "value_in_quotes" or null,
        "source": "value_in_quotes" or null,
        "destination": "value_in_quotes" or null,
        "protocol": "value_in_quotes" or null,
        "port": "value_in_quotes" or null,
        "format": "table"
    }}
}}

**CRITICAL RULES:**
1. All string values MUST be in double quotes. Extract the EXACT value from the CURRENT USER QUERY, not from examples.
2. The tool_name MUST be one of the exact tool names listed above. Do NOT invent tool names like "query_panorama_object_groups" - use "query_panorama_ip_object_group" exactly.
3. If you are unsure which tool to use, return needs_clarification: true with a clarification question.
4. **If the query requests an action that NO available tool can perform (e.g., "create an address object", "delete a device", "modify configuration"), return tool_name: null, needs_clarification: false, and set clarification_question to explain that this system cannot perform that action.**

**EXAMPLE - Query that cannot be processed:**
Current query: "create an address object named ravi_k with 12.0.0.0/24 in it"
Correct response:
{{"tool_name": null, "needs_clarification": false, "clarification_question": "I'm sorry, but this system is not equipped to create or modify network configuration objects like address objects. I can only query existing information such as network paths, device locations, rack details, and Panorama address group memberships. Would you like to query information about an existing address object instead?", "parameters": {{"ip_address": null, "device_name": null, "rack_name": null, "address_group_name": null, "device_group": null, "site_name": null, "intent": null, "format": "table"}}}}

**EXAMPLE - Query that cannot be processed (modification):**
Current query: "delete device roundrock-sw-1"
Correct response:
{{"tool_name": null, "needs_clarification": false, "clarification_question": "I'm sorry, but this system is not equipped to delete or modify network devices. I can only query existing information such as network paths, device locations, rack details, and Panorama address group memberships. Would you like to look up information about this device instead?", "parameters": {{"ip_address": null, "device_name": null, "rack_name": null, "address_group_name": null, "device_group": null, "site_name": null, "intent": null, "format": "table"}}}}
"""
    return prompt_text


async def select_tool_with_llm(
    prompt: str,
    tools_description: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    llm_model: str = "llama3.1:8b",
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
