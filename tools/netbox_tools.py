"""
NetBox domain module -- MCP tool functions for querying NetBox.

Provides tools for rack details, rack listing, and device rack-location
lookups.  All tools are registered on the shared ``mcp`` FastMCP instance
imported from ``tools.shared``.
"""

import sys
import ssl
import json
import asyncio
import re
import aiohttp
from typing import Optional, Dict, Any, List

from tools.shared import (
    mcp,
    _get_llm,
    NETBOX_URL,
    NETBOX_TOKEN,
    NETBOX_VERIFY_SSL,
    ChatPromptTemplate,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _netbox_headers() -> Dict[str, str]:
    """Build NetBox API headers."""
    headers = {
        "Accept": "application/json"
    }
    if NETBOX_TOKEN:
        headers["Authorization"] = f"Token {NETBOX_TOKEN}"
    return headers


def _netbox_ssl_context() -> Optional[ssl.SSLContext]:
    """Return SSL context for NetBox if verification is disabled."""
    if NETBOX_VERIFY_SSL:
        return None
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context


def _device_manufacturer(device: Dict[str, Any]) -> Any:
    """Extract manufacturer from a NetBox device (device_type.manufacturer or device.manufacturer)."""
    dt = device.get("device_type")
    if isinstance(dt, dict):
        mfr = dt.get("manufacturer")
        if isinstance(mfr, dict):
            return mfr.get("display") or mfr.get("name")
        return mfr
    mfr = device.get("manufacturer")
    if isinstance(mfr, dict):
        return mfr.get("display") or mfr.get("name")
    return mfr


async def _fetch_elevation_image_base64(elevation_url: str) -> Optional[str]:
    """
    Fetch NetBox elevation SVG from API and return as base64-encoded image.

    Args:
        elevation_url: URL to NetBox elevation page (web UI format)

    Returns:
        Base64-encoded SVG string (data:image/svg+xml;base64,...) or None if failed
    """
    if not elevation_url:
        return None

    try:
        # Extract rack_id from URL: /dcim/racks/{rack_id}/elevation/?face=front
        import re
        rack_id_match = re.search(r'/racks/(\d+)/', elevation_url)
        if not rack_id_match:
            print(f"DEBUG: Could not extract rack_id from URL: {elevation_url}", file=sys.stderr, flush=True)
            return None

        rack_id = rack_id_match.group(1)

        # Extract face parameter
        face_match = re.search(r'face=(\w+)', elevation_url)
        face = face_match.group(1) if face_match else 'front'

        # Use NetBox API endpoint that returns SVG directly
        api_url = f"{NETBOX_URL}/api/dcim/racks/{rack_id}/elevation/?face={face}&render=svg"

        headers = _netbox_headers()
        headers["Accept"] = "image/svg+xml"
        ssl_context = _netbox_ssl_context()

        print(f"DEBUG: Fetching elevation SVG from API: {api_url}", file=sys.stderr, flush=True)

        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, headers=headers, ssl=ssl_context, timeout=15) as response:
                if response.status == 200:
                    svg_content = await response.text()

                    # Convert SVG to base64 data URI
                    import base64
                    svg_base64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
                    print(f"DEBUG: Successfully fetched SVG ({len(svg_content)} bytes)", file=sys.stderr, flush=True)
                    return f"data:image/svg+xml;base64,{svg_base64}"
                else:
                    error_text = await response.text()
                    print(f"DEBUG: API returned status {response.status}: {error_text[:200]}", file=sys.stderr, flush=True)
                    return None

    except Exception as e:
        print(f"DEBUG: Error fetching elevation SVG: {e}", file=sys.stderr, flush=True)
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return None


# ---------------------------------------------------------------------------
# MCP tool functions
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_rack_details(
    rack_name: str,
    site_name: Optional[str] = None,
    format: Optional[str] = None,
    conversation_history: Optional[str] = None  # JSON string instead of List[Dict]
) -> Dict[str, Any]:
    """
    Use ONLY for rack name lookup (e.g. "A4", "A1"). Input: rack_name (short, no dots). NOT for "path allowed" or "traffic allowed" with two IPs—use check_path_allowed. Rack names have no dots; IPs have dots.

    Use this tool when the user asks about a specific rack by SHORT name (A4, A1, B2). Do not use for device names with dashes (get_device_rack_location).

    When to use this tool:
    - Query contains a RACK NAME: short identifier like "A1", "A4", "B2", "Rack A4"
    - **CRITICAL: If query mentions "space utilization", "utilization", "rack details", "rack" with a SHORT name (like "A4") → this is ALWAYS a rack query → use this tool**
    - **ABSOLUTE RULE #1: If the query is JUST a short identifier like "A4", "A1", "B2" (1-3 characters, letter + number, no dashes, no dots) → this is ALWAYS a rack name → use this tool IMMEDIATELY. DO NOT return tool_name as null.**
    - **ABSOLUTE RULE #2: DO NOT ask for clarification when the query is clearly a rack name like "A4" - just use this tool directly with rack_name="A4"**
    - **ABSOLUTE RULE #3: If you see a query that is just "A4" or "A1" or any short identifier matching the rack name pattern, you MUST return tool_name="get_rack_details" with rack_name set to that value. Never return null for tool_name in this case.**

    **HANDLING FOLLOW-UP RESPONSES:**
    - If conversation history shows a previous clarification question was asked in the standard format: "What would you like to do with [IP]? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path"
    - AND the current query is just "3" or "three" → this means the user selected option 3 (Look up rack in NetBox)
    - **CRITICAL: The standard clarification question order is: 1) Panorama, 2) Device, 3) Rack, 4) Network Path - if you see "3" and the question lists "3) Look up rack in NetBox", use this tool**
    - Extract the rack name or IP address from EARLIER messages in the conversation history (the message immediately before the clarification question)
    - Use this tool (get_rack_details) with the rack name from history
    - Example: User says "11.0.0.1" → clarification asked with "1) Query Panorama..., 2) Look up device..., 3) Look up rack in NetBox..." → user responds "3" → use this tool with rack_name from history (if rack name exists) or IP address

    **Rack name identification:**
    - Rack names are SHORT (1-3 characters)
    - Rack names do NOT contain dashes (-)
    - Rack names do NOT contain dots (.) - IP addresses have dots, so they are NOT rack names
    - Pattern: letter(s) + number(s), e.g., "A1", "A4", "B12"
    - If you see "A1" or "A4" → this is a RACK NAME → use this tool
    - If you see "11.0.0.1" or any string with dots → this is an IP ADDRESS, NOT a rack name → do NOT use this tool
    - If you see ANY name with DASHES (e.g., "roundrock-dc-border-leaf1", "leander-dc-leaf1") → this is a DEVICE NAME → use get_device_rack_location instead

    **IMPORTANT: Do NOT confuse with device names:**
    - Device names ALWAYS contain DASHES (e.g., "roundrock-dc-border-leaf1", "leander-dc-leaf1") → use get_device_rack_location
    - Rack names NEVER contain dashes (e.g., "A1", "A4") → use this tool (get_rack_details)
    - CRITICAL RULE: If a name contains a dash (-), it is ALWAYS a device name, NEVER a rack name!
    - CRITICAL: "roundrock-dc-border-leaf1" has dashes = DEVICE, NOT a rack!
    - CRITICAL: "leander-dc-leaf1" has dashes = DEVICE, NOT a rack!
    - CRITICAL: "border-leaf1" is PART of a device name, NOT a rack name!

    **Examples:**
    - Query: "rack details for A4" → rack_name="A4", site_name=None, format="table"
    - Query: "rack details of A4" → rack_name="A4", site_name=None, format="table"
    - Query: "A4" → rack_name="A4", site_name=None, format="table"
    - Query: "show me rack A4" → rack_name="A4", site_name=None, format="table"
    - Query: "A1 in Round Rock DC" → rack_name="A1", site_name="Round Rock DC", format="table"
    - Query: "show rack A4" → rack_name="A4", site_name=None, format="table"
    - Query: "A1" → rack_name="A1", site_name=None, format="table"
    - Query: "space utilization of A4" → rack_name="A4", site_name=None, format="table" (A4 is a rack name, NOT a device!)
    - Query: "rack A4 utilization" → rack_name="A4", site_name=None, format="table"
    - Query: "A4 space usage" → rack_name="A4", site_name=None, format="table"

    **Query variations (all → get_rack_details with rack_name=short id; use site_name if location given):**
    - "rack A4" / "rack A1" / "A4" / "A1" / "show rack A4" / "show me rack A1"
    - "rack details for A4" / "rack details of A1" / "details for rack A4" / "info on rack A1"
    - "A4 in Leander" / "A4 at Leander DC" / "rack A1 in Round Rock" / "A1 at Round Rock DC"
    - "space utilization of A4" / "A4 utilization" / "A4 space usage" / "rack A4 utilization"
    - "which devices are in rack A4?" / "devices in A4" / "what's in rack A4"
    - Any phrase with a SHORT rack id (A1, A4, B2, no dashes) → use this tool. Names with DASHES are devices → use get_device_rack_location instead.

    The tool can return data in different formats:
    - table: Returns data formatted as a table (recommended)
    - json: Returns data in JSON format
    - list: Returns data as a list
    - None: Returns a natural language summary with AI analysis

    Args:
        rack_name: The SHORT rack identifier (e.g., "A4", "A1", "B2") - must be short, no dashes
        site_name: Optional site name to filter racks (e.g., "Round Rock DC", "Leander DC")
        format: Output format - "table" (recommended), "json", "list", or None for natural language summary
        conversation_history: Optional conversation history for context-aware responses

    Returns:
        dict: Rack details including rack name, site, location, units, devices in rack, and AI-generated summary
    """
    # Parse conversation_history if it's a JSON string
    if isinstance(conversation_history, str):
        try:
            conversation_history = json.loads(conversation_history) if conversation_history else None
        except json.JSONDecodeError:
            conversation_history = None

    rack_name = (rack_name or "").strip()
    if not rack_name:
        return {"error": "Rack name cannot be empty"}

    # CRITICAL: Reject IP addresses (contain dots) - path/traffic allowed queries must use check_path_allowed
    if "." in rack_name:
        return {
            "error": f"'{rack_name}' looks like an IP address (contains dots), not a rack name. Rack names are short (e.g. A4, A1).",
            "suggestion": "For queries like 'is path allowed from 10.0.0.1 to 10.0.1.1?' or 'is traffic allowed?' use the check_path_allowed tool with source and destination IPs."
        }

    # CRITICAL: Reject device names (names with dashes) - they should use get_device_rack_location instead
    if "-" in rack_name:
        return {
            "error": f"'{rack_name}' is a device name (contains dashes), not a rack name. Use get_device_rack_location tool instead.",
            "suggestion": "Device names contain dashes (e.g., 'leander-dc-leaf5'). Rack names are short identifiers without dashes (e.g., 'A1', 'A4')."
        }

    if not NETBOX_TOKEN:
        return {"error": "NETBOX_TOKEN is not set. Configure NetBox API token."}

    # Clean rack name: remove "Rack" prefix and any location suffixes
    # Examples: "Rack A4" -> "A4", "A4 in Leander" -> "A4", "rack details for A4" -> "A4"
    rack_name_clean = rack_name.replace("Rack", "").replace("rack", "").strip()
    # Remove location suffixes like "in Leander", "at site", etc.
    import re
    rack_name_clean = re.sub(r'\s+(in|at|for|from)\s+[A-Za-z\s]+$', '', rack_name_clean, flags=re.IGNORECASE).strip()
    # Remove "details" if present
    rack_name_clean = re.sub(r'\s+details\s*$', '', rack_name_clean, flags=re.IGNORECASE).strip()

    # If site_name is provided, first get the site ID
    site_id = None
    site_name_found = None
    if site_name:
        site_name_clean = site_name.strip()
        sites_url = f"{NETBOX_URL}/api/dcim/sites/"
        async with aiohttp.ClientSession() as session:
            try:
                # Try exact name match first
                site_params = {"name": site_name_clean}
                async with session.get(sites_url, headers=_netbox_headers(), params=site_params, ssl=_netbox_ssl_context(), timeout=15) as response:
                    if response.status == 200:
                        site_data = await response.json()
                        site_results = site_data.get("results", []) if isinstance(site_data, dict) else []
                        if site_results:
                            # Try exact match first (case-insensitive)
                            for site in site_results:
                                site_name_lower = str(site.get("name", "")).lower()
                                if site_name_lower == site_name_clean.lower():
                                    site_id = site.get("id")
                                    site_name_found = site.get("name")
                                    break

                            # If no exact match, try partial/fuzzy match
                            if not site_id and site_results:
                                site_name_clean_lower = site_name_clean.lower()
                                # Remove common suffixes for matching
                                site_name_clean_lower_no_suffix = site_name_clean_lower.replace(" dc", "").replace(" data center", "").replace(" datacenter", "")

                                for site in site_results:
                                    site_display = str(site.get("name", "")).lower()
                                    site_display_no_suffix = site_display.replace(" dc", "").replace(" data center", "").replace(" datacenter", "")

                                    # Check if cleaned names match
                                    if (site_name_clean_lower_no_suffix in site_display_no_suffix or
                                        site_display_no_suffix in site_name_clean_lower_no_suffix or
                                        site_name_clean_lower in site_display or
                                        site_display in site_name_clean_lower):
                                        site_id = site.get("id")
                                        site_name_found = site.get("name")
                                        break

                            # Last resort: use first result if we have any
                            if not site_id and site_results:
                                site_id = site_results[0].get("id")
                                site_name_found = site_results[0].get("name")

                        # If no results with exact name, try search query
                        if not site_id:
                            search_params = {"q": site_name_clean}
                            async with session.get(sites_url, headers=_netbox_headers(), params=search_params, ssl=_netbox_ssl_context(), timeout=15) as response:
                                if response.status == 200:
                                    search_data = await response.json()
                                    search_results = search_data.get("results", []) if isinstance(search_data, dict) else []
                                    if search_results:
                                        # Try to find best match
                                        site_name_clean_lower = site_name_clean.lower()
                                        for site in search_results:
                                            site_display = str(site.get("name", "")).lower()
                                            if site_name_clean_lower in site_display or site_display in site_name_clean_lower:
                                                site_id = site.get("id")
                                                site_name_found = site.get("name")
                                                break
                                        if not site_id and search_results:
                                            site_id = search_results[0].get("id")
                                            site_name_found = search_results[0].get("name")
            except Exception as e:
                print(f"DEBUG: Error looking up site: {str(e)}", file=sys.stderr, flush=True)

    url = f"{NETBOX_URL}/api/dcim/racks/"
    headers = _netbox_headers()
    ssl_context = _netbox_ssl_context()

    async with aiohttp.ClientSession() as session:
        try:
            # Use search (q=) first so we get ALL racks matching this name across sites; paginate to avoid assuming one DC
            all_results = []
            params = {"q": rack_name_clean, "limit": 250}
            if site_id:
                params["site_id"] = site_id
            next_url = url
            first_params = params
            async with session.get(url, headers=headers, params=first_params, ssl=ssl_context, timeout=15) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {
                        "error": f"NetBox rack lookup failed: HTTP {response.status}",
                        "details": error_text[:500]
                    }
                data = await response.json()
                results_page = data.get("results", []) if isinstance(data, dict) else []
                all_results.extend(results_page)
                next_url = data.get("next") if isinstance(data, dict) else None
                print(f"DEBUG get_rack_details: API response keys={list(data.keys()) if isinstance(data, dict) else 'n/a'}, count={data.get('count') if isinstance(data, dict) else 'n/a'}, results_this_page={len(results_page)}, next={bool(next_url)}", file=sys.stderr, flush=True)
                if results_page:
                    r0 = results_page[0]
                    site_raw = r0.get("site")
                    print(f"DEBUG get_rack_details: first result name={r0.get('name')}, site type={type(site_raw).__name__}, site value={site_raw}", file=sys.stderr, flush=True)
            while next_url:
                async with session.get(next_url, headers=headers, ssl=ssl_context, timeout=15) as response:
                    if response.status != 200:
                        break
                    data = await response.json()
                    all_results.extend(data.get("results", []) if isinstance(data, dict) else [])
                    next_url = data.get("next") if isinstance(data, dict) else None
            results = all_results
            print(f"DEBUG get_rack_details: total results after pagination={len(results)}", file=sys.stderr, flush=True)
            if not results:
                # Fallback: exact name filter in case search (q) returned nothing
                params_exact = {"name": rack_name_clean, "limit": 250}
                if site_id:
                    params_exact["site_id"] = site_id
                async with session.get(url, headers=headers, params=params_exact, ssl=ssl_context, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", []) if isinstance(data, dict) else []
        except Exception as e:
            return {"error": f"NetBox rack lookup error: {str(e)}"}

        if not results:
            return {
                "rack": rack_name_clean,
                "error": "Rack not found in NetBox"
            }

        # Collect all racks with matching name
        matching_racks = []
        for candidate in results:
            if str(candidate.get("name", "")).lower() == rack_name_clean.lower():
                matching_racks.append(candidate)

        # If no exact matches, use first result
        if not matching_racks:
            matching_racks = [results[0]] if results else []
        print(f"DEBUG get_rack_details: matching_racks count={len(matching_racks)}, site_id={site_id}", file=sys.stderr, flush=True)

        # If site filtering was requested, find rack in that site
        rack = None
        if site_id:
            for candidate in matching_racks:
                rack_site = candidate.get("site")
                rack_site_id = rack_site.get("id") if isinstance(rack_site, dict) else rack_site
                if rack_site_id == site_id:
                    rack = candidate
                    break

        # If no site filter or no match with site filter, check for multiple sites
        if not rack:
            # Get unique sites from matching racks (resolve site name if API returned only ID)
            sites = {}
            for candidate in matching_racks:
                site = candidate.get("site") or {}
                site_id_val = site.get("id") if isinstance(site, dict) else site
                site_name_val = site.get("name") or site.get("display") if isinstance(site, dict) else site
                if site_id_val is None:
                    continue
                if site_id_val not in sites:
                    if site_name_val and str(site_name_val).strip() and not str(site_name_val).isdigit():
                        sites[site_id_val] = str(site_name_val).strip()
                    else:
                        # Resolve site name by ID so we can show "Leander DC" not "Site ID 1"
                        try:
                            site_url = f"{NETBOX_URL}/api/dcim/sites/{site_id_val}/"
                            async with session.get(site_url, headers=headers, ssl=ssl_context, timeout=5) as site_resp:
                                if site_resp.status == 200:
                                    site_data = await site_resp.json()
                                    sites[site_id_val] = (site_data.get("name") or site_data.get("display") or f"Site ID {site_id_val}").strip()
                                else:
                                    sites[site_id_val] = f"Site ID {site_id_val}"
                        except Exception:
                            sites[site_id_val] = f"Site ID {site_id_val}"
            print(f"DEBUG get_rack_details: unique sites={sites}, len(sites)={len(sites)}", file=sys.stderr, flush=True)

            # If user provided site_name but we couldn't resolve site_id (e.g. "Leander" vs "Leander DC"),
            # match against the site names we already have from the matching racks
            if site_name and not site_id and sites:
                site_name_clean_lower = site_name.strip().lower()
                site_name_no_suffix = site_name_clean_lower.replace(" dc", "").replace(" data center", "").replace(" datacenter", "")
                matched_site_id = None
                for sid, sname in sites.items():
                    sname_lower = str(sname).lower()
                    sname_no_suffix = sname_lower.replace(" dc", "").replace(" data center", "").replace(" datacenter", "")
                    if (site_name_clean_lower == sname_lower or
                        site_name_clean_lower in sname_lower or sname_lower in site_name_clean_lower or
                        site_name_no_suffix == sname_no_suffix or
                        site_name_no_suffix in sname_no_suffix or sname_no_suffix in site_name_no_suffix):
                        matched_site_id = sid
                        break
                if matched_site_id is not None:
                    site_id = matched_site_id
                    for candidate in matching_racks:
                        rack_site = candidate.get("site")
                        rid = rack_site.get("id") if isinstance(rack_site, dict) else rack_site
                        if rid == site_id:
                            rack = candidate
                            break
                    print(f"DEBUG get_rack_details: matched user site_name to site id {site_id}, rack={rack.get('name') if rack else None}", file=sys.stderr, flush=True)

            # If multiple sites and no site filter provided (and we didn't just resolve above), ask which site
            if not rack and len(sites) > 1 and not site_id:
                return {
                    "rack": rack_name_clean,
                    "error": f"Multiple racks named '{rack_name_clean}' found at different sites",
                    "sites": list(sites.values()),
                    "requires_site": True
                }

            # Use first matching rack if still not set
            if not rack:
                rack = matching_racks[0]

        # Get site information
        site = rack.get("site") or {}
        site_name = site.get("name") or site.get("display") if isinstance(site, dict) else site

        # Get location information
        location = rack.get("location") or {}
        location_name = location.get("name") or location.get("display") if isinstance(location, dict) else location

        # Get devices in this rack
        devices_url = f"{NETBOX_URL}/api/dcim/devices/"
        devices_in_rack = []
        try:
            params = {"rack_id": rack.get("id")}
            async with session.get(devices_url, headers=headers, params=params, ssl=ssl_context, timeout=15) as response:
                if response.status == 200:
                    devices_data = await response.json()
                    devices_in_rack = devices_data.get("results", []) if isinstance(devices_data, dict) else []
        except Exception as e:
            print(f"DEBUG: Error fetching devices in rack: {str(e)}", file=sys.stderr, flush=True)

        # Calculate space utilization
        u_height = rack.get("u_height") or 42  # Default to 42U if not specified
        occupied_units = 0
        device_positions = set()

        for device in devices_in_rack:
            position = device.get("position")
            if position:
                # Get device height (device_type.u_height), default to 1U if not specified
                device_type = device.get("device_type", {})
                if isinstance(device_type, dict):
                    device_u_height = device_type.get("u_height") or 1
                else:
                    device_u_height = 1

                # Count occupied units (handle devices that span multiple U)
                for u in range(int(position), int(position) + int(device_u_height)):
                    if u not in device_positions:
                        device_positions.add(u)
                        occupied_units += 1

        space_utilization = (occupied_units / u_height * 100) if u_height > 0 else 0

        # Get rack ID for elevation URLs
        rack_id = rack.get("id")

        # Build result
        result = {
            "rack": rack.get("name") or rack_name_clean,
            "site": site_name,
            "location": location_name,
            "facility_id": rack.get("facility_id"),
            "status": rack.get("status", {}).get("value") if isinstance(rack.get("status"), dict) else rack.get("status"),
            "role": rack.get("role", {}).get("name") if isinstance(rack.get("role"), dict) else rack.get("role"),
            "type": rack.get("type", {}).get("value") if isinstance(rack.get("type"), dict) else rack.get("type"),
            "width": rack.get("width", {}).get("value") if isinstance(rack.get("width"), dict) else rack.get("width"),
            "u_height": rack.get("u_height"),
            "desc_units": rack.get("desc_units"),
            "outer_width": rack.get("outer_width"),
            "outer_depth": rack.get("outer_depth"),
            "outer_unit": rack.get("outer_unit", {}).get("value") if isinstance(rack.get("outer_unit"), dict) else rack.get("outer_unit"),
            "space_utilization": round(space_utilization, 1),
            "occupied_units": occupied_units,
            "devices_count": len(devices_in_rack),
            "devices": [
                {
                    "name": device.get("name"),
                    "position": device.get("position"),
                    "face": device.get("face", {}).get("value") if isinstance(device.get("face"), dict) else device.get("face"),
                    "device_type": device.get("device_type", {}).get("display") if isinstance(device.get("device_type"), dict) else device.get("device_type"),
                    "status": device.get("status", {}).get("value") if isinstance(device.get("status"), dict) else device.get("status"),
                    "manufacturer": _device_manufacturer(device),
                    "device_role": device.get("device_role", {}).get("display") if isinstance(device.get("device_role"), dict) else device.get("device_role"),
                    "serial": device.get("serial"),
                    "primary_ip": device.get("primary_ip", {}).get("address") if isinstance(device.get("primary_ip"), dict) else device.get("primary_ip"),
                }
                for device in devices_in_rack
            ]
        }

        # Add elevation URLs and try to fetch images
        if rack_id:
            # NetBox elevation URLs: /dcim/racks/{id}/elevation/?face=front or /dcim/racks/{id}/elevation/?face=rear
            # Use trailing slash format (NetBox standard)
            result["elevation_front_url"] = f"{NETBOX_URL}/dcim/racks/{rack_id}/elevation/?face=front"
            result["elevation_rear_url"] = f"{NETBOX_URL}/dcim/racks/{rack_id}/elevation/?face=rear"

            # Try to fetch elevation images as base64
            try:
                front_img_base64 = await _fetch_elevation_image_base64(result["elevation_front_url"])
                if front_img_base64:
                    result["elevation_front_image"] = front_img_base64

                rear_img_base64 = await _fetch_elevation_image_base64(result["elevation_rear_url"])
                if rear_img_base64:
                    result["elevation_rear_image"] = rear_img_base64
            except Exception as e:
                print(f"DEBUG: Error fetching elevation images: {e}", file=sys.stderr, flush=True)

        # Try to enhance with LLM analysis if available
        llm = _get_llm()
        if llm is not None:
            try:
                device_details = {
                    "rack": result["rack"],
                    "site": result["site"],
                    "location": result["location"],
                    "facility_id": result["facility_id"],
                    "status": result["status"],
                    "role": result["role"],
                    "type": result["type"],
                    "width": result["width"],
                    "u_height": result["u_height"],
                    "space_utilization": result.get("space_utilization"),
                    "occupied_units": result.get("occupied_units"),
                    "devices_count": result["devices_count"],
                    "devices": result["devices"]
                }
                device_details = {k: v for k, v in device_details.items() if v is not None}

                format_instruction = ""
                if format == "table":
                    format_instruction = "\n\nIMPORTANT: The user requested the response in TABLE FORMAT. Format your summary as a markdown table with columns: Field | Value."
                elif format == "json":
                    format_instruction = "\n\nIMPORTANT: The user requested JSON format. Ensure your response is properly structured JSON."
                elif format == "list":
                    format_instruction = "\n\nIMPORTANT: The user requested list format. Format information as bullet points or numbered lists."

                conversation_context = ""
                if conversation_history and len(conversation_history) > 0:
                    conv_text = "\n".join([
                        f"{msg.get('role', 'unknown').title()}: {msg.get('content', '')}"
                        for msg in conversation_history[-10:]
                    ])
                    conversation_context = f"\n\nCONVERSATION CONTEXT:\n{conv_text}\n\nUse this conversation history to understand the user's intent and provide contextually relevant responses."

                system_prompt = f"""You are a network infrastructure assistant. Analyze the rack information from NetBox and provide a concise summary.

Provide a summary that includes:
- Rack name and location (site, facility ID if available)
- Rack specifications (type, width, height in U, status)
- Space utilization percentage (if available)
- Number of devices in the rack
- Key devices and their positions if relevant

Focus on factual information from the rack data. Keep the summary concise and informative.
{format_instruction}
{conversation_context}

Format your response as a JSON object with this field:
{{
    "summary": "string - concise summary of the rack information"
}}"""

                analysis_prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "Analyze this rack information from NetBox:\n{{rack_data}}")
                ])

                formatted_messages = analysis_prompt_template.format_messages(
                    format_instruction=format_instruction,
                    conversation_context=conversation_context,
                    rack_data=json.dumps(device_details, indent=2)
                )

                print(f"DEBUG: Invoking LLM for rack analysis", file=sys.stderr, flush=True)
                response = await asyncio.wait_for(
                    llm.ainvoke(formatted_messages),
                    timeout=30.0
                )
                content = response.content if hasattr(response, 'content') else str(response)

                print(f"DEBUG: LLM response received: {content[:200]}...", file=sys.stderr, flush=True)

                # Extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group())
                        result["ai_analysis"] = {
                            "summary": analysis.get("summary", content)
                        }
                    except json.JSONDecodeError:
                        result["ai_analysis"] = {"summary": content}
                else:
                    result["ai_analysis"] = {"summary": content}

            except Exception as e:
                print(f"DEBUG: LLM analysis failed: {str(e)}", file=sys.stderr, flush=True)
                result["ai_analysis"] = {"summary": f"Rack {result['rack']} located at {result.get('site', 'Unknown site')} with {result['devices_count']} devices."}

        return result


@mcp.tool()
async def list_racks(
    site_name: Optional[str] = None,
    format: Optional[str] = None,
    conversation_history: Optional[str] = None  # JSON string instead of List[Dict]
) -> Dict[str, Any]:
    """
    List all racks from NetBox, optionally filtered by site.

    **CRITICAL: When to use this tool:**
    - Use this tool when the query asks for "all racks", "list racks", "show racks", "racks in [site]"
    - Use this tool when the query asks for racks at a specific site (e.g., "racks in Leander DC", "racks at Round Rock")
    - Use this tool when the query asks for "all racks" without specifying a particular rack name
    - Do NOT use this tool if the query contains a specific rack name (like "A4", "A1") - use get_rack_details instead

    **Examples:**
    - Query: "list all racks" → site_name=None, format="table"
    - Query: "show all racks" → site_name=None, format="table"
    - Query: "racks in Leander DC" → site_name="Leander DC", format="table"
    - Query: "racks at Round Rock" → site_name="Round Rock", format="table"
    - Query: "what racks are in Leander" → site_name="Leander", format="table"
    - Query: "how many racks are in round rock" → site_name="round rock", format="table"
    - Query: "which racks are at Round Rock" → site_name="Round Rock", format="table"
    - Query: "all racks in Round Rock DC" → site_name="Round Rock DC", format="table"

    **Query variations (all → list_racks; set site_name when a location is mentioned):**
    - "list all racks" / "show all racks" / "all racks" / "list racks" / "show racks"
    - "what racks are in Leander" / "what racks are in leander" / "which racks are in Round Rock"
    - "how many racks are in round rock" / "how many racks in Leander DC"
    - "racks at Leander" / "racks in Leander DC" / "racks at Round Rock" / "racks in Round Rock DC"
    - "list racks at Leander" / "list racks in Round Rock" / "show racks at Leander DC"
    - "give me all racks at Leander" / "show me racks in Round Rock" / "get racks at Leander DC"
    - "count of racks in round rock" / "number of racks at Leander"
    - "which racks are at Round Rock?" / "what racks are at Leander DC?"
    - Do NOT use for a single rack name (e.g. "A4", "A1") → use get_rack_details instead.

    The tool can return data in different formats:
    - table: Returns data formatted as a table (recommended)
    - json: Returns data in JSON format
    - list: Returns data as a list
    - None: Returns a natural language summary with AI analysis

    Args:
        site_name: Optional site name to filter racks (e.g., "Round Rock DC", "Leander DC"). If None, returns all racks.
        format: Output format - "table" (recommended), "json", "list", or None for natural language summary
        conversation_history: Optional conversation history for context-aware responses

    Returns:
        dict: List of racks with their details (name, site, status, space utilization, device count, etc.)
    """
    # Parse conversation_history if it's a JSON string
    if isinstance(conversation_history, str):
        try:
            conversation_history = json.loads(conversation_history) if conversation_history else None
        except json.JSONDecodeError:
            conversation_history = None

    if not NETBOX_TOKEN:
        return {"error": "NETBOX_TOKEN is not set. Configure NetBox API token."}

    # If site_name is provided, first get the site ID
    site_id = None
    site_name_found = None
    if site_name:
        site_name_clean = site_name.strip()
        sites_url = f"{NETBOX_URL}/api/dcim/sites/"
        async with aiohttp.ClientSession() as session:
            try:
                # Try exact name match first
                site_params = {"name": site_name_clean}
                async with session.get(sites_url, headers=_netbox_headers(), params=site_params, ssl=_netbox_ssl_context(), timeout=15) as response:
                    if response.status == 200:
                        site_data = await response.json()
                        site_results = site_data.get("results", []) if isinstance(site_data, dict) else []
                        if site_results:
                            # Try exact match first (case-insensitive)
                            for site in site_results:
                                site_name_lower = str(site.get("name", "")).lower()
                                if site_name_lower == site_name_clean.lower():
                                    site_id = site.get("id")
                                    site_name_found = site.get("name")
                                    break

                            # If no exact match, try partial/fuzzy match
                            if not site_id and site_results:
                                site_name_clean_lower = site_name_clean.lower()
                                # Remove common suffixes for matching
                                site_name_clean_lower_no_suffix = site_name_clean_lower.replace(" dc", "").replace(" data center", "").replace(" datacenter", "")

                                for site in site_results:
                                    site_display = str(site.get("name", "")).lower()
                                    site_display_no_suffix = site_display.replace(" dc", "").replace(" data center", "").replace(" datacenter", "")

                                    # Check if cleaned names match
                                    if (site_name_clean_lower_no_suffix in site_display_no_suffix or
                                        site_display_no_suffix in site_name_clean_lower_no_suffix or
                                        site_name_clean_lower in site_display or
                                        site_display in site_name_clean_lower):
                                        site_id = site.get("id")
                                        site_name_found = site.get("name")
                                        break

                            # Last resort: use first result if we have any
                            if not site_id and site_results:
                                site_id = site_results[0].get("id")
                                site_name_found = site_results[0].get("name")

                        # If no results with exact name, try search query
                        if not site_id:
                            search_params = {"q": site_name_clean}
                            async with session.get(sites_url, headers=_netbox_headers(), params=search_params, ssl=_netbox_ssl_context(), timeout=15) as response:
                                if response.status == 200:
                                    search_data = await response.json()
                                    search_results = search_data.get("results", []) if isinstance(search_data, dict) else []
                                    if search_results:
                                        # Try to find best match
                                        site_name_clean_lower = site_name_clean.lower()
                                        for site in search_results:
                                            site_display = str(site.get("name", "")).lower()
                                            if site_name_clean_lower in site_display or site_display in site_name_clean_lower:
                                                site_id = site.get("id")
                                                site_name_found = site.get("name")
                                                break
                                        if not site_id and search_results:
                                            site_id = search_results[0].get("id")
                                            site_name_found = search_results[0].get("name")
            except Exception as e:
                print(f"DEBUG: Error looking up site: {str(e)}", file=sys.stderr, flush=True)

    # Get racks from NetBox
    url = f"{NETBOX_URL}/api/dcim/racks/"
    headers = _netbox_headers()
    ssl_context = _netbox_ssl_context()

    racks_list = []
    async with aiohttp.ClientSession() as session:
        try:
            params = {}
            if site_id:
                params["site_id"] = site_id

            # Get all racks (with pagination if needed)
            all_racks = []
            while True:
                async with session.get(url, headers=headers, params=params, ssl=ssl_context, timeout=15) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "error": f"NetBox rack lookup failed: HTTP {response.status}",
                            "details": error_text[:500]
                        }
                    data = await response.json()
                    results = data.get("results", []) if isinstance(data, dict) else []
                    all_racks.extend(results)

                    # Check for next page
                    next_url = data.get("next")
                    if not next_url:
                        break
                    # Extract offset/limit from next_url or use next page
                    # For simplicity, we'll just get the first page for now
                    # In production, you'd want to handle pagination properly
                    break

            # Process each rack to get space utilization
            for rack in all_racks:
                rack_id = rack.get("id")
                rack_name = rack.get("name")
                site = rack.get("site") or {}
                site_name_rack = site.get("name") or site.get("display") if isinstance(site, dict) else site

                # Get devices in this rack to calculate space utilization
                devices_url = f"{NETBOX_URL}/api/dcim/devices/"
                devices_in_rack = []
                try:
                    device_params = {"rack_id": rack_id}
                    async with session.get(devices_url, headers=headers, params=device_params, ssl=ssl_context, timeout=15) as response:
                        if response.status == 200:
                            devices_data = await response.json()
                            devices_in_rack = devices_data.get("results", []) if isinstance(devices_data, dict) else []
                except Exception as e:
                    print(f"DEBUG: Error fetching devices for rack {rack_name}: {str(e)}", file=sys.stderr, flush=True)

                # Calculate space utilization
                u_height = rack.get("u_height") or 42
                occupied_units = 0
                device_positions = set()

                for device in devices_in_rack:
                    position = device.get("position")
                    if position:
                        device_type = device.get("device_type", {})
                        if isinstance(device_type, dict):
                            device_u_height = device_type.get("u_height") or 1
                        else:
                            device_u_height = 1

                        for u in range(int(position), int(position) + int(device_u_height)):
                            if u not in device_positions:
                                device_positions.add(u)
                                occupied_units += 1

                space_utilization = (occupied_units / u_height * 100) if u_height > 0 else 0

                elevation_front_url = f"{NETBOX_URL}/dcim/racks/{rack_id}/elevation/?face=front" if rack_id else None
                elevation_rear_url = f"{NETBOX_URL}/dcim/racks/{rack_id}/elevation/?face=rear" if rack_id else None

                # Try to fetch elevation images
                elevation_front_image = None
                elevation_rear_image = None
                if rack_id:
                    try:
                        elevation_front_image = await _fetch_elevation_image_base64(elevation_front_url)
                        elevation_rear_image = await _fetch_elevation_image_base64(elevation_rear_url)
                    except Exception as e:
                        print(f"DEBUG: Error fetching elevation images for rack {rack_name}: {e}", file=sys.stderr, flush=True)

                racks_list.append({
                    "rack": rack_name,
                    "rack_id": rack_id,  # Add rack_id for elevation URLs
                    "site": site_name_rack,
                    "status": rack.get("status", {}).get("value") if isinstance(rack.get("status"), dict) else rack.get("status"),
                    "u_height": rack.get("u_height"),
                    "space_utilization": round(space_utilization, 1),
                    "occupied_units": occupied_units,
                    "devices_count": len(devices_in_rack),
                    "elevation_front_url": elevation_front_url,
                    "elevation_rear_url": elevation_rear_url,
                    "elevation_front_image": elevation_front_image,
                    "elevation_rear_image": elevation_rear_image
                })

            result = {
                "racks": racks_list,
                "total_count": len(racks_list),
                "site_filter": site_name_found if site_name else None
            }

            # Try to enhance with LLM analysis if available
            llm = _get_llm()
            if llm is not None:
                try:
                    format_instruction = ""
                    if format == "table":
                        format_instruction = "\n\nIMPORTANT: The user requested the response in TABLE FORMAT. Format your summary as a markdown table."
                    elif format == "json":
                        format_instruction = "\n\nIMPORTANT: The user requested JSON format. Ensure your response is properly structured JSON."
                    elif format == "list":
                        format_instruction = "\n\nIMPORTANT: The user requested list format. Format information as bullet points or numbered lists."

                    conversation_context = ""
                    if conversation_history and len(conversation_history) > 0:
                        conv_text = "\n".join([
                            f"{msg.get('role', 'unknown').title()}: {msg.get('content', '')}"
                            for msg in conversation_history[-10:]
                        ])
                        conversation_context = f"\n\nCONVERSATION CONTEXT:\n{conv_text}\n\nUse this conversation history to understand the user's intent and provide contextually relevant responses."

                    site_filter_text = f" at {site_name_found}" if site_name_found else ""
                    system_prompt = f"""You are a network infrastructure assistant. Analyze the list of racks from NetBox and provide a concise summary.

Provide a summary that includes:
- Total number of racks{site_filter_text}
- Key information about the racks (sites, space utilization, device counts)
- Any notable patterns or insights

Focus on factual information from the rack data. Keep the summary concise and informative.
{format_instruction}
{conversation_context}

Format your response as a JSON object with this field:
{{
    "summary": "string - concise summary of the racks information"
}}"""

                    analysis_prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "Analyze this list of racks from NetBox:\n{{racks_data}}")
                    ])

                    formatted_messages = analysis_prompt_template.format_messages(
                        format_instruction=format_instruction,
                        conversation_context=conversation_context,
                        racks_data=json.dumps(racks_list, indent=2)
                    )

                    print(f"DEBUG: Invoking LLM for racks list analysis", file=sys.stderr, flush=True)
                    response = await asyncio.wait_for(
                        llm.ainvoke(formatted_messages),
                        timeout=30.0
                    )
                    content = response.content if hasattr(response, 'content') else str(response)

                    print(f"DEBUG: LLM response received: {content[:200]}...", file=sys.stderr, flush=True)

                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            analysis = json.loads(json_match.group())
                            result["ai_analysis"] = {
                                "summary": analysis.get("summary", content)
                            }
                        except json.JSONDecodeError:
                            result["ai_analysis"] = {"summary": content}
                    else:
                        result["ai_analysis"] = {"summary": content}

                except Exception as e:
                    print(f"DEBUG: LLM analysis failed: {str(e)}", file=sys.stderr, flush=True)
                    result["ai_analysis"] = {"summary": f"Found {len(racks_list)} rack(s){' at ' + site_name_found if site_name_found else ''}."}

            return result

        except Exception as e:
            return {"error": f"NetBox rack lookup error: {str(e)}"}


@mcp.tool()
async def get_device_rack_location(
    device_name: str,
    intent: Optional[str] = None,
    format: Optional[str] = None,
    expected_rack: Optional[str] = None,  # For yes/no questions like "is device X in rack Y?"
    conversation_history: Optional[str] = None  # JSON string instead of List[Dict]
) -> Dict[str, Any]:
    """
    Use ONLY for NetBox device lookup by device name (e.g. leander-dc-leaf1). Input: device_name (names have dashes). NOT for "path allowed" or "traffic allowed" between two IPs—use check_path_allowed. Queries with dots (IPs) are not device names.

    **QUICK CHECK: Does the user query contain a DOT (.)?**
    - YES (e.g., "11.0.0.1", "192.168.1.1") → This is an IP address → DO NOT use this tool → Ask for clarification instead
    - NO (e.g., "leander-dc-leaf6", "roundrock-dc-leaf1") → This might be a device name → Continue reading below

    **ABSOLUTE RULE #1: If the user query is JUST an IP address (like "11.0.0.1", "11.0.0.2", "192.168.1.1") with NO other words, you MUST ask for clarification. DO NOT select this tool.**

    **ABSOLUTE RULE #2: IP addresses have DOTS (.) - Device names have DASHES (-). They are completely different.**
    - "11.0.0.1" has dots → it's an IP address → DO NOT use this tool → ask for clarification
    - "leander-dc-leaf6" has dashes → it's a device name → you CAN use this tool

    **ABSOLUTE RULE #3: This tool requires a device_name parameter. If the query is just an IP address, there is NO device_name, so DO NOT use this tool.**

    **CRITICAL: This tool is ONLY for DEVICE NAMES, NOT for IP addresses.**

    **CRITICAL: When to use this tool:**
    - Use this tool ONLY when the query contains a DEVICE NAME (a name with DASHES, e.g., "roundrock-dc-border-leaf1", "leander-dc-leaf1")
    - Device names are LONG strings with multiple dashes separating parts
    - Examples of device names: "roundrock-dc-border-leaf1", "leander-dc-border-leaf2", "roundrock-dc-leaf1"

    **CRITICAL: When NOT to use this tool - READ CAREFULLY:**
    - **DO NOT use this tool for IP addresses** (e.g., "11.0.0.1", "11.0.0.2", "192.168.1.100")
    - **IP addresses have DOTS (.), device names have DASHES (-) - they are completely different**
    - If the query is JUST an IP address (like "11.0.0.1" or "11.0.0.2") with no device name:
      * Ask for clarification using the standard format: "What would you like to do with [IP]? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path"
      * OR use query_panorama_ip_object_group if the query explicitly mentions Panorama/object groups
    - **NEVER use this tool for queries that are just IP addresses - IP addresses are NOT device names**
    - Example: "11.0.0.1" is an IP address (has dots) → do NOT use this tool, ask for clarification
    - Example: "11.0.0.2" is an IP address (has dots) → do NOT use this tool, ask for clarification
    - Example: "roundrock-dc-leaf1" is a device name (has dashes) → use this tool

    **HANDLING FOLLOW-UP RESPONSES:**
    - If conversation history shows a previous clarification question was asked in the standard format: "What would you like to do with [IP]? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path"
    - AND the current query is just "2" or "two" → this means the user selected option 2 (Look up device in NetBox)
    - **CRITICAL: Only use this tool (get_device_rack_location) when user responds "2" to a clarification question that lists "2) Look up device in NetBox"**
    - **CRITICAL: The standard clarification question order is: 1) Panorama, 2) Device, 3) Rack, 4) Network Path - if you see "2" and the question lists "2) Look up device in NetBox", use this tool**
    - Extract the IP address or device name from EARLIER messages in the conversation history (the message immediately before the clarification question)
    - Use this tool (get_device_rack_location) with the device name or IP from history
    - **IMPORTANT: When using this tool for a follow-up response, if you extracted an IP address, pass it as device_name parameter (NetBox can look up devices by IP)**
    - Example: User says "11.0.0.1" → clarification asked with "1) Query Panorama..., 2) Look up device in NetBox..." → user responds "2" → use this tool with device_name="11.0.0.1" (from history, even though it's an IP, pass it as device_name)
    - **DO NOT use this tool when user responds "1" to a clarification question - that means Panorama query (option 1)**

    **Device name identification:**
    - Device names ALWAYS contain DASHES (-)
    - Device names are typically 15+ characters long
    - If you see "roundrock-dc-border-leaf1" → this is a DEVICE NAME → use this tool
    - If you see "A1" or "A4" → this is a RACK NAME → use get_rack_details instead
    - **CRITICAL: If you see "11.0.0.1" or any string with DOTS (.) → this is an IP ADDRESS, NOT a device name → do NOT use this tool**
    - IP addresses have dots (.), device names have dashes (-) - they are completely different

    **Intent parameter (what the user wants to see):**
    - "device_details" (default): Show all device information (rack, position, site, status, device type, manufacturer, model, etc.)
    - "rack_location_only": Show only rack location (site, rack, position) - use when query contains "rack location" or "where is"
    - "device_type_only": Show only device type - use when query contains "device type"
    - "status_only": Show only device status - use when query contains "status"
    - "site_only": Show only site name - use when query contains "site" or "what site" or "which site" (e.g., "site leander-dc-leaf1" → intent="site_only")
    - "manufacturer_only": Show only manufacturer - use when query contains "manufacturer"

    **CRITICAL: Query format examples:**
    - "site leander-dc-leaf1" → device_name="leander-dc-leaf1", intent="site_only" (NOT a rack query!)
    - "what site does leander-dc-leaf1 belong to" → device_name="leander-dc-leaf1", intent="site_only"
    - "leander-dc-leaf1" (just device name) → device_name="leander-dc-leaf1", intent="device_details"

    **Format parameter:**
    - "table" (recommended): Returns data formatted as a table
    - "json": Returns data in JSON format
    - "list": Returns data as a list
    - None: Returns a natural language summary with AI analysis

    **Examples:**
    - Query: "roundrock-dc-border-leaf1" → device_name="roundrock-dc-border-leaf1", intent="device_details", format="table"
    - Query: "rack location roundrock-dc-border-leaf1" → device_name="roundrock-dc-border-leaf1", intent="rack_location_only", format="table"
    - Query: "manufacturer of roundrock-dc-border-leaf1" → device_name="roundrock-dc-border-leaf1", intent="manufacturer_only", format="table"
    - Query: "status of roundrock-dc-border-leaf1" → device_name="roundrock-dc-border-leaf1", intent="status_only", format="table"

    **Query variations (all → get_device_rack_location with device_name=the name with dashes; this is NetBox device lookup, NOT Panorama):**
    - "is leander-dc-border-leaf1 in A1?" / "is leander-dc-border-leaf1 racked in A1?" / "is device X in rack Y?" → device_name="leander-dc-border-leaf1" (look up device; result shows which rack it is in—compare to A1)
    - "on netbox where is leander-dc-border-leaf1?" / "in netbox where is leander-dc-border-leaf1?"
    - "where is leander-dc-border-leaf1?" / "where is leander-dc-border-leaf1 racked?"
    - "which rack is leander-dc-border-leaf1 in?" / "what rack is leander-dc-border-leaf1 in?"
    - "rack location of leander-dc-border-leaf1" / "rack location for leander-dc-border-leaf1"
    - "find leander-dc-border-leaf1 in netbox" / "look up leander-dc-border-leaf1 in netbox"
    - "netbox where is leander-dc-border-leaf1" / "netbox find device leander-dc-border-leaf1"
    - "locate leander-dc-border-leaf1" / "locate device leander-dc-border-leaf1"
    - "which site is leander-dc-border-leaf1 in?" / "what site is leander-dc-border-leaf1 in?" → intent="site_only"
    - "device type of leander-dc-border-leaf1" → intent="device_type_only"
    - Any phrase with a name containing DASHES (e.g. roundrock-dc-leaf1, leander-dc-border-leaf1) asking where/location/rack/site → use this tool. Do NOT use Panorama tools for device names with dashes.

    Args:
        device_name: The FULL device name to look up (e.g., "roundrock-dc-border-leaf1" - must include all parts with dashes).
                     **CRITICAL: This parameter accepts ONLY device names (strings with DASHES like "leander-dc-leaf6"), NOT IP addresses (strings with DOTS like "11.0.0.1").**
                     **If you have an IP address, DO NOT use this tool - ask for clarification instead.**
        intent: What information to return - "device_details" (all info), "rack_location_only", "device_type_only", "status_only", "site_only", "manufacturer_only"
        format: Output format - "table" (recommended), "json", "list", or None for natural language summary
        conversation_history: Optional conversation history for context-aware responses

    Returns:
        dict: Device information based on intent - all details, or specific field(s) as requested
    """
    # Parse conversation_history if it's a JSON string
    if isinstance(conversation_history, str):
        try:
            conversation_history = json.loads(conversation_history) if conversation_history else None
        except json.JSONDecodeError:
            conversation_history = None

    device_name = (device_name or "").strip()
    if not device_name:
        return {"error": "Device name cannot be empty"}
    if not NETBOX_TOKEN:
        return {"error": "NETBOX_TOKEN is not set. Configure NetBox API token."}

    url = f"{NETBOX_URL}/api/dcim/devices/"
    headers = _netbox_headers()
    ssl_context = _netbox_ssl_context()

    async with aiohttp.ClientSession() as session:
        try:
            # Try exact name match first
            params = {"name": device_name}
            async with session.get(url, headers=headers, params=params, ssl=ssl_context, timeout=15) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {
                        "error": f"NetBox device lookup failed: HTTP {response.status}",
                        "details": error_text[:500]
                    }
                data = await response.json()
        except Exception as e:
            return {"error": f"NetBox device lookup error: {str(e)}"}

        results = data.get("results", []) if isinstance(data, dict) else []
        if not results:
            # Fallback to generic search
            try:
                params = {"q": device_name}
                async with session.get(url, headers=headers, params=params, ssl=ssl_context, timeout=15) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "error": f"NetBox device search failed: HTTP {response.status}",
                            "details": error_text[:500]
                        }
                    data = await response.json()
                    results = data.get("results", []) if isinstance(data, dict) else []
            except Exception as e:
                return {"error": f"NetBox device search error: {str(e)}"}

        if not results:
            return {
                "device": device_name,
                "rack": None,
                "message": "Device not found in NetBox"
            }

        # Prefer exact name match if present
        device = None
        for candidate in results:
            if str(candidate.get("name", "")).lower() == device_name.lower():
                device = candidate
                break
        if device is None:
            device = results[0]

        rack = device.get("rack") or {}
        site = device.get("site") or rack.get("site") or {}
        rack_name = rack.get("name") or rack.get("display") or rack.get("id")
        site_name = site.get("name") or site.get("display") or site.get("id")
        face_value = device.get("face") or device.get("rack_face")
        if isinstance(face_value, dict):
            face_value = face_value.get("label") or face_value.get("value")
        if not isinstance(face_value, str):
            face_value = None

        # Build basic result with location info and device details
        # Default intent is "device_details" (show all)
        intent = intent or "device_details"

        # Build full result first
        full_result = {
            "device": device.get("name") or device_name,
            "rack": rack_name,
            "position": device.get("position") or device.get("rack_position"),
            "face": face_value,
            "site": site_name,
            "status": (device.get("status") or {}).get("value"),
            # Include additional device details for table display
            "device_type": device.get("device_type", {}).get("display") if isinstance(device.get("device_type"), dict) else device.get("device_type"),
            "device_role": device.get("device_role", {}).get("display") if isinstance(device.get("device_role"), dict) else device.get("device_role"),
            # Manufacturer is stored in device_type.manufacturer in NetBox
            "manufacturer": (
                device.get("device_type", {}).get("manufacturer", {}).get("display")
                if isinstance(device.get("device_type"), dict) and isinstance(device.get("device_type", {}).get("manufacturer"), dict)
                else (device.get("device_type", {}).get("manufacturer")
                      if isinstance(device.get("device_type"), dict)
                      else (device.get("manufacturer", {}).get("display")
                            if isinstance(device.get("manufacturer"), dict)
                            else device.get("manufacturer")))
            ),
            "model": device.get("model", {}).get("display") if isinstance(device.get("model"), dict) else device.get("model"),
            "serial": device.get("serial"),
            "primary_ip": device.get("primary_ip", {}).get("address") if isinstance(device.get("primary_ip"), dict) else device.get("primary_ip"),
            "primary_ip4": device.get("primary_ip4", {}).get("address") if isinstance(device.get("primary_ip4"), dict) else device.get("primary_ip4"),
            "intent": intent,  # Store intent in result for client display logic
        }

        # Filter result based on intent
        if intent == "rack_location_only":
            result = {
                "device": full_result["device"],
                "rack": full_result["rack"],
                "position": full_result["position"],
                "site": full_result["site"],
                "intent": intent,
            }
        elif intent == "device_type_only":
            result = {
                "device": full_result["device"],
                "device_type": full_result["device_type"],
                "intent": intent,
            }
        elif intent == "status_only":
            result = {
                "device": full_result["device"],
                "status": full_result["status"],
                "intent": intent,
            }
        elif intent == "site_only":
            result = {
                "device": full_result["device"],
                "site": full_result["site"],
                "intent": intent,
            }
        elif intent == "manufacturer_only":
            result = {
                "device": full_result["device"],
                "manufacturer": full_result["manufacturer"],
                "intent": intent,
            }
        else:  # device_details or default
            result = full_result

        # Try to enhance with LLM analysis if available (lazy initialization)
        llm = _get_llm()
        result["_debug_llm_check"] = {
            "hasattr_llm": hasattr(mcp, 'llm'),
            "llm_value": str(mcp.llm),
            "llm_is_none": llm is None,
            "llm_type": str(type(llm)) if llm else "None",
            "llm_error": getattr(mcp, 'llm_error', None)
        }

        if llm is not None:
            print(f"DEBUG: LLM available, starting analysis for rack location", file=sys.stderr, flush=True)
            result["_debug_llm_check"]["status"] = "LLM available, attempting analysis"
            try:
                print(f"DEBUG: Entering LLM analysis try block", file=sys.stderr, flush=True)
                # Get full device details from NetBox for LLM analysis
                # The device object contains all NetBox fields
                print(f"DEBUG: Preparing device_details from device object", file=sys.stderr, flush=True)
                device_details = {
                    "name": device.get("name"),
                    "display": device.get("display"),
                    "device_type": device.get("device_type", {}).get("display") if isinstance(device.get("device_type"), dict) else device.get("device_type"),
                    "device_role": device.get("device_role", {}).get("display") if isinstance(device.get("device_role"), dict) else device.get("device_role"),
                    # Manufacturer is stored in device_type, not directly on device
                    "manufacturer": (
                        device.get("device_type", {}).get("manufacturer", {}).get("display")
                        if isinstance(device.get("device_type"), dict) and isinstance(device.get("device_type", {}).get("manufacturer"), dict)
                        else (device.get("device_type", {}).get("manufacturer")
                              if isinstance(device.get("device_type"), dict)
                              else (device.get("manufacturer", {}).get("display")
                                    if isinstance(device.get("manufacturer"), dict)
                                    else device.get("manufacturer")))
                    ),
                    "model": device.get("model", {}).get("display") if isinstance(device.get("model"), dict) else device.get("model"),
                    "serial": device.get("serial"),
                    "asset_tag": device.get("asset_tag"),
                    "rack": device.get("rack"),
                    "position": device.get("position"),
                    "face": device.get("face"),
                    "site": device.get("site"),
                    "location": device.get("location"),
                    "tenant": device.get("tenant"),
                    "platform": device.get("platform"),
                    "status": device.get("status"),
                    "primary_ip": device.get("primary_ip"),
                    "primary_ip4": device.get("primary_ip4"),
                    "primary_ip6": device.get("primary_ip6"),
                    "cluster": device.get("cluster"),
                    "virtual_chassis": device.get("virtual_chassis"),
                    "vc_position": device.get("vc_position"),
                    "vc_priority": device.get("vc_priority"),
                    "comments": device.get("comments"),
                    "tags": device.get("tags"),
                    "custom_fields": device.get("custom_fields"),
                }
                # Remove None values to keep JSON clean
                device_details = {k: v for k, v in device_details.items() if v is not None}
                print(f"DEBUG: Device details prepared, keys: {list(device_details.keys())}", file=sys.stderr, flush=True)

                # Use LangChain ChatPromptTemplate for structured prompt management
                print(f"DEBUG: Building format instruction and conversation context", file=sys.stderr, flush=True)
                format_instruction = ""
                if format == "table":
                    format_instruction = "\n\nIMPORTANT: The user requested the response in TABLE FORMAT. Format your summary and notes as a markdown table with columns: Field | Value. Structure the information clearly in table rows."
                elif format == "json":
                    format_instruction = "\n\nIMPORTANT: The user requested JSON format. Ensure your response is properly structured JSON."
                elif format == "list":
                    format_instruction = "\n\nIMPORTANT: The user requested list format. Format information as bullet points or numbered lists."

                # Build conversation context if available
                conversation_context = ""
                if conversation_history and len(conversation_history) > 0:
                    # Format conversation history for context
                    conv_text = "\n".join([
                        f"{msg.get('role', 'unknown').title()}: {msg.get('content', '')}"
                        for msg in conversation_history[-10:]  # Last 10 messages for context
                    ])
                    conversation_context = f"\n\nCONVERSATION CONTEXT:\n{conv_text}\n\nUse this conversation history to understand the user's intent and provide contextually relevant responses."

                # Build the system prompt - use double braces {{ }} to escape them in format strings
                # LangChain's format_messages uses .format() internally, so we need to escape braces
                system_prompt = """You are a network infrastructure assistant. Analyze the complete device information from NetBox and provide:
Provide a concise summary that MUST start with the rack location in this EXACT format: "Device [device name] located at [Site name] - Rack [rack name], Position U[integer], Face [face]. [Manufacturer] [device type/model], [status]."

CRITICAL FORMATTING RULES:
- Site and Rack are DIFFERENT: Site is the data center/location name, Rack is the physical rack name (e.g., "A4")
- Position MUST be formatted as U[integer] with NO decimals (e.g., "U1" NOT "U1.0" - always remove decimals and convert to integer)
- Face should be lowercase (e.g., "front" not "Front")
- ALWAYS include Manufacturer if it exists in the device data (e.g., "Arista", "Cisco", etc.)
- Format: Manufacturer name followed by device type/model
- Status should be included at the end

Example format: "Device leander-dc-border-leaf1 located at Leander DC - Rack A4, Position U1, Face front. Arista DCS-7050SX3-24YC4C-S-F, active."

Focus on factual information from the device data. Keep the summary concise and informative.
{format_instruction}
{conversation_context}

Format your response as a JSON object with this field:
{{
    "summary": "string - MUST start with: Device [name] located at [Site] - Rack [rack], Position U[integer], Face [face]. [details]"
}}"""

                analysis_prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "Analyze this device information from NetBox:\n{device_data}")
                ])

                # Format the prompt with the full device data
                print(f"DEBUG: Formatting prompt template with device data", file=sys.stderr, flush=True)
                formatted_messages = analysis_prompt_template.format_messages(
                    format_instruction=format_instruction,
                    conversation_context=conversation_context,
                    device_data=json.dumps(device_details, indent=2, default=str)
                )
                print(f"DEBUG: Prompt formatted, about to invoke LLM", file=sys.stderr, flush=True)

                print(f"DEBUG: Invoking LLM for rack location analysis...", file=sys.stderr, flush=True)
                # Invoke the LLM with the formatted prompt (with timeout)
                try:
                    llm_response = await asyncio.wait_for(
                        llm.ainvoke(formatted_messages),
                        timeout=30.0  # 30 second timeout
                    )
                    print(f"DEBUG: LLM response received", file=sys.stderr, flush=True)
                except asyncio.TimeoutError:
                    print(f"DEBUG: LLM invocation timed out after 30 seconds", file=sys.stderr, flush=True)
                    raise
                except Exception as llm_invoke_error:
                    print(f"DEBUG: LLM invocation failed: {str(llm_invoke_error)}", file=sys.stderr, flush=True)
                    import traceback
                    print(f"DEBUG: LLM invocation traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                    raise

                # Extract content from the response
                if hasattr(llm_response, 'content'):
                    response_content = llm_response.content
                else:
                    response_content = str(llm_response)

                print(f"DEBUG: LLM response content length: {len(response_content)}", file=sys.stderr, flush=True)
                print(f"DEBUG: LLM response preview: {response_content[:500]}", file=sys.stderr, flush=True)

                # Simple approach: try to find and extract JSON from the response
                import re
                json_content = None
                analysis = None

                # First, try to extract JSON from markdown code blocks
                json_block_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response_content, re.DOTALL)
                if json_block_match:
                    json_content = json_block_match.group(1).strip()
                    print(f"DEBUG: Extracted JSON from markdown code block", file=sys.stderr, flush=True)
                else:
                    # Try to find JSON object by matching balanced braces
                    start_idx = response_content.find('{')
                    if start_idx != -1:
                        brace_count = 0
                        for i in range(start_idx, len(response_content)):
                            if response_content[i] == '{':
                                brace_count += 1
                            elif response_content[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_content = response_content[start_idx:i+1]
                                    print(f"DEBUG: Extracted JSON object from text", file=sys.stderr, flush=True)
                                    break

                # Try to parse the JSON - wrap in try-except to catch any KeyError
                analysis = None
                try:
                    if json_content:
                        try:
                            analysis = json.loads(json_content)
                            # Verify it's a dict and has valid keys
                            if isinstance(analysis, dict):
                                # Log original keys for debugging
                                original_keys = list(analysis.keys())
                                print(f"DEBUG: Original JSON keys (before cleaning): {[repr(k) for k in original_keys]}", file=sys.stderr, flush=True)

                                # Check if keys are valid (no quotes, spaces, or newlines)
                                valid_keys = {}
                                for k, v in analysis.items():
                                    # Ensure key is a clean string - strip ALL whitespace including newlines
                                    original_key = str(k)
                                    clean_key = original_key.strip()
                                    # Remove quotes if present (handles both single and double quotes)
                                    if clean_key.startswith('"') and clean_key.endswith('"'):
                                        clean_key = clean_key[1:-1].strip()
                                    elif clean_key.startswith("'") and clean_key.endswith("'"):
                                        clean_key = clean_key[1:-1].strip()
                                    # Remove any remaining whitespace/newlines from the key
                                    clean_key = clean_key.replace('\n', '').replace('\r', '').replace('\t', '').strip()
                                    # Only add if key is not empty after cleaning
                                    if clean_key:
                                        valid_keys[clean_key] = v
                                        if original_key != clean_key:
                                            print(f"DEBUG: Cleaned key: {repr(original_key)} -> {repr(clean_key)}", file=sys.stderr, flush=True)
                                    else:
                                        print(f"DEBUG: Skipping empty key after cleaning: {repr(k)}", file=sys.stderr, flush=True)
                                analysis = valid_keys
                                print(f"DEBUG: Successfully parsed JSON, cleaned keys: {list(analysis.keys())}", file=sys.stderr, flush=True)
                            else:
                                analysis = {"summary": str(analysis)[:1000]}
                        except json.JSONDecodeError as je:
                            print(f"DEBUG: JSON parse failed: {str(je)}", file=sys.stderr, flush=True)
                            # Try to extract just the summary field using regex
                            summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', response_content, re.DOTALL)
                            if summary_match:
                                analysis = {"summary": summary_match.group(1)}
                                print(f"DEBUG: Extracted summary using regex", file=sys.stderr, flush=True)
                            else:
                                # Last resort: use the full response as summary
                                analysis = {"summary": response_content[:1000]}
                                print(f"DEBUG: Using full response as summary", file=sys.stderr, flush=True)
                    else:
                        # No JSON found, extract summary from text
                        summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', response_content, re.DOTALL)
                        if summary_match:
                            analysis = {"summary": summary_match.group(1)}
                            print(f"DEBUG: Extracted summary from text (no JSON block found)", file=sys.stderr, flush=True)
                        else:
                            analysis = {"summary": response_content[:1000]}
                            print(f"DEBUG: Using full response as summary (no JSON found)", file=sys.stderr, flush=True)
                except KeyError as ke:
                    print(f"DEBUG: KeyError during JSON parsing/extraction: {str(ke)}", file=sys.stderr, flush=True)
                    analysis = {"summary": response_content[:1000] if 'response_content' in locals() else "Analysis error: KeyError in JSON parsing"}
                except Exception as parse_err:
                    print(f"DEBUG: Error during JSON parsing/extraction: {str(parse_err)}", file=sys.stderr, flush=True)
                    analysis = {"summary": response_content[:1000] if 'response_content' in locals() else f"Analysis error: {str(parse_err)[:200]}"}


                # SIMPLIFIED: Just ensure analysis is a dict with a summary field
                # Wrap everything in try-except to catch any KeyError
                try:
                    # First, ensure analysis exists and is a dict
                    if analysis is None:
                        analysis = {"summary": "Analysis unavailable - LLM response was empty"}
                    elif not isinstance(analysis, dict):
                        analysis = {"summary": str(analysis)[:1000] if analysis else "Analysis unavailable"}

                    # Now safely extract summary - use only .get() to avoid KeyError
                    summary = None
                    # Try standard keys first using .get() (never use 'in' operator which might fail with malformed keys)
                    summary = analysis.get("summary") or analysis.get("Summary") or analysis.get("SUMMARY")

                    # If still not found, try case-insensitive search with aggressive cleaning
                    if not summary:
                        try:
                            keys_list = list(analysis.keys())  # Convert to list first
                            for k in keys_list:
                                try:
                                    # Aggressively clean the key: remove all whitespace, newlines, quotes
                                    k_str = str(k).replace('\n', '').replace('\r', '').replace('\t', '').strip()
                                    # Remove quotes if present
                                    if k_str.startswith('"') and k_str.endswith('"'):
                                        k_str = k_str[1:-1].strip()
                                    elif k_str.startswith("'") and k_str.endswith("'"):
                                        k_str = k_str[1:-1].strip()
                                    k_str = k_str.lower()
                                    if k_str == "summary":
                                        summary = analysis.get(k)  # Use .get() - should never raise KeyError
                                        if summary:
                                            print(f"DEBUG: Found summary using cleaned key: {repr(k)} -> 'summary'", file=sys.stderr, flush=True)
                                            break
                                except Exception as key_err:
                                    print(f"DEBUG: Error processing key {repr(k)}: {str(key_err)}", file=sys.stderr, flush=True)
                                    continue
                        except Exception as search_err:
                            print(f"DEBUG: Error in case-insensitive search: {str(search_err)}", file=sys.stderr, flush=True)
                            pass

                    # If still no summary, use first value
                    if not summary:
                        try:
                            if analysis:
                                summary = next(iter(analysis.values()), None)
                        except Exception:
                            pass

                    # Create simple result dict with only summary
                    final_analysis = {"summary": str(summary)[:1000] if summary else "Analysis completed"}

                    result["ai_analysis"] = final_analysis
                    print(f"DEBUG: Added ai_analysis to result with keys: {list(final_analysis.keys())}", file=sys.stderr, flush=True)
                except KeyError as ke:
                    # Catch KeyError here and create fallback
                    print(f"DEBUG: KeyError in analysis processing: {str(ke)}", file=sys.stderr, flush=True)
                    import traceback
                    print(f"DEBUG: KeyError traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                    result["ai_analysis"] = {"summary": response_content[:1000] if 'response_content' in locals() else "Analysis error: KeyError occurred"}
                except Exception as analysis_err:
                    # Catch any other error
                    print(f"DEBUG: Error in analysis processing: {str(analysis_err)}", file=sys.stderr, flush=True)
                    import traceback
                    print(f"DEBUG: Analysis error traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                    result["ai_analysis"] = {"summary": response_content[:1000] if 'response_content' in locals() else f"Analysis error: {str(analysis_err)[:200]}"}
            except asyncio.TimeoutError as te:
                # Log timeout error
                print(f"DEBUG: LLM analysis timed out: {str(te)}", file=sys.stderr, flush=True)
                result["_debug_llm_check"]["llm_timeout"] = True
                error_msg = str(te).replace('\n', ' ').replace('\r', ' ')[:100]
                result["_debug_llm_check"]["llm_timeout_error"] = error_msg
            except KeyError as ke:
                # Handle KeyError specifically
                error_msg = f"KeyError: {str(ke)}"
                print(f"DEBUG: LLM analysis KeyError: {error_msg}", file=sys.stderr, flush=True)
                import traceback
                print(f"DEBUG: KeyError traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                # Try to log what keys are available if analysis exists
                if 'analysis' in locals() and isinstance(analysis, dict):
                    print(f"DEBUG: Available keys in analysis dict: {[repr(k) for k in analysis.keys()]}", file=sys.stderr, flush=True)
                result["_debug_llm_check"]["llm_error"] = error_msg[:150]
                result["_debug_llm_check"]["llm_error_type"] = "KeyError"
                # Try to create a basic analysis even on KeyError
                try:
                    # Use the raw response as summary if available
                    if 'response_content' in locals():
                        result["ai_analysis"] = {"summary": response_content[:500]}
                        print(f"DEBUG: Created fallback ai_analysis from response_content", file=sys.stderr, flush=True)
                    elif 'llm_response' in locals():
                        # Try to get content from llm_response
                        try:
                            if hasattr(llm_response, 'content'):
                                content = str(llm_response.content)[:500]
                                result["ai_analysis"] = {"summary": content}
                                print(f"DEBUG: Created fallback ai_analysis from llm_response.content", file=sys.stderr, flush=True)
                        except:
                            result["ai_analysis"] = {"summary": "LLM analysis encountered a KeyError. Check server logs for details."}
                            print(f"DEBUG: Created basic fallback ai_analysis", file=sys.stderr, flush=True)
                    else:
                        result["ai_analysis"] = {"summary": "LLM analysis encountered a KeyError. Check server logs for details."}
                        print(f"DEBUG: Created basic fallback ai_analysis", file=sys.stderr, flush=True)
                except Exception as fallback_error:
                    print(f"DEBUG: Failed to create fallback ai_analysis: {str(fallback_error)}", file=sys.stderr, flush=True)
                    # Last resort: create a minimal analysis
                    result["ai_analysis"] = {"summary": "Analysis unavailable due to error"}
            except Exception as e:
                # Log the error for debugging but don't fail the entire request
                print(f"DEBUG: LLM analysis failed for rack location: {str(e)}", file=sys.stderr, flush=True)
                print(f"DEBUG: Exception type: {type(e).__name__}", file=sys.stderr, flush=True)
                import traceback
                traceback_str = traceback.format_exc()
                print(f"DEBUG: LLM analysis traceback: {traceback_str}", file=sys.stderr, flush=True)
                # Store error info safely (ensure it's JSON-serializable)
                try:
                    # Sanitize error message: remove newlines, control characters, and limit length
                    error_str = str(e)
                    # First, replace all escape sequences and control characters
                    error_str = error_str.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
                    error_str = error_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    # Remove all non-printable characters except spaces
                    error_str = ''.join(char for char in error_str if char.isprintable() or char == ' ')
                    # Collapse multiple spaces
                    error_str = ' '.join(error_str.split())
                    # Remove any remaining problematic characters
                    error_str = error_str.replace('"', "'").replace('\\', '/')
                    # Limit length
                    error_str = error_str[:150]
                    result["_debug_llm_check"]["llm_error"] = error_str
                    result["_debug_llm_check"]["llm_error_type"] = type(e).__name__
                    print(f"DEBUG: Stored sanitized error: {error_str}", file=sys.stderr, flush=True)
                except Exception as store_error:
                    print(f"DEBUG: Failed to store error info: {str(store_error)}", file=sys.stderr, flush=True)
                    # Use a very safe fallback
                    result["_debug_llm_check"]["llm_error"] = f"Error type: {type(e).__name__}"
                    result["_debug_llm_check"]["llm_error_type"] = type(e).__name__
                pass
        else:
            print(f"DEBUG: LLM not available (llm={llm})", file=sys.stderr, flush=True)
            result["_debug_llm_check"]["status"] = f"LLM not available (llm={llm})"

        print(f"DEBUG: Final result keys: {list(result.keys())}", file=sys.stderr, flush=True)
        print(f"DEBUG: Final result has ai_analysis: {'ai_analysis' in result}", file=sys.stderr, flush=True)

        # Clean up result to ensure it's JSON-serializable
        # Remove any non-serializable values and sanitize error messages
        def clean_value(v):
            """Recursively clean a value to ensure it's JSON-serializable."""
            if v is None:
                return None
            elif isinstance(v, (str, int, float, bool)):
                # For strings, ensure they're safe
                if isinstance(v, str):
                    # Remove control characters and problematic characters
                    # First handle escape sequences
                    v = v.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
                    v = v.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    # Remove all non-printable characters except spaces
                    v = ''.join(char for char in v if char.isprintable() or char == ' ')
                    # Collapse multiple spaces
                    v = ' '.join(v.split())
                    # Limit length for very long strings
                    if len(v) > 5000:
                        v = v[:5000] + "... (truncated)"
                return v
            elif isinstance(v, dict):
                # Safely iterate over dict items, handling any key errors
                cleaned_dict = {}
                try:
                    for k, val in v.items():
                        try:
                            # Ensure key is a string
                            key_str = str(k) if not isinstance(k, str) else k
                            cleaned_dict[key_str] = clean_value(val)
                        except Exception as key_error:
                            print(f"DEBUG: Error cleaning dict key '{k}': {str(key_error)}", file=sys.stderr, flush=True)
                            # Skip problematic keys
                            continue
                except Exception as dict_error:
                    print(f"DEBUG: Error iterating dict: {str(dict_error)}", file=sys.stderr, flush=True)
                    # Return string representation if we can't iterate
                    return str(v)[:500]
                return cleaned_dict
            elif isinstance(v, (list, tuple)):
                return [clean_value(item) for item in v]
            else:
                # Try to convert to string
                try:
                    str_val = str(v)
                    # Sanitize string representation
                    str_val = str_val.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
                    str_val = str_val.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    str_val = ''.join(char for char in str_val if char.isprintable() or char == ' ')
                    str_val = ' '.join(str_val.split())
                    if len(str_val) > 500:
                        str_val = str_val[:500] + "... (truncated)"
                    return str_val
                except:
                    return f"<non-serializable: {type(v).__name__}>"

        cleaned_result = {}
        for key, value in result.items():
            try:
                cleaned_result[key] = clean_value(value)
                # Test if it's JSON-serializable
                json.dumps(cleaned_result[key], default=str)
            except (TypeError, ValueError) as e:
                print(f"DEBUG: WARNING - Key '{key}' has non-serializable value after cleaning, using fallback: {str(e)}", file=sys.stderr, flush=True)
                cleaned_result[key] = f"<error serializing: {type(value).__name__}>"

        # Ensure result is JSON-serializable (FastMCP will serialize it)
        try:
            # Test JSON serialization to catch any issues
            json_str = json.dumps(cleaned_result, default=str, ensure_ascii=False)
            print(f"DEBUG: Result is JSON-serializable, length: {len(json_str)}", file=sys.stderr, flush=True)
            # Verify ai_analysis is in the serialized JSON
            if "ai_analysis" in json_str:
                print(f"DEBUG: ai_analysis found in JSON serialization", file=sys.stderr, flush=True)
            else:
                print(f"DEBUG: WARNING - ai_analysis NOT found in JSON serialization!", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"DEBUG: ERROR - Result is NOT JSON-serializable: {str(e)}", file=sys.stderr, flush=True)
            import traceback
            print(f"DEBUG: JSON serialization traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
            # Return a safe fallback result
            return {
                "device": result.get("device", "unknown"),
                "rack": result.get("rack"),
                "position": result.get("position"),
                "site": result.get("site"),
                "status": result.get("status"),
                "error": "Result serialization failed",
                "_debug_llm_check": result.get("_debug_llm_check", {})
            }

        # Add yes/no answer if expected_rack is provided (for questions like "is device X in rack Y?")
        print(f"DEBUG: expected_rack parameter = {repr(expected_rack)}", file=sys.stderr, flush=True)
        if expected_rack:
            print(f"DEBUG: Generating yes/no answer for expected_rack={expected_rack}", file=sys.stderr, flush=True)
            actual_rack = cleaned_result.get("rack")
            device_name_str = cleaned_result.get("device", device_name)

            # Normalize rack names for comparison (remove spaces, case-insensitive)
            expected_normalized = str(expected_rack).strip().upper() if expected_rack else ""
            actual_normalized = str(actual_rack).strip().upper() if actual_rack else ""

            if actual_normalized == expected_normalized:
                cleaned_result["yes_no_answer"] = f"\u2705 **Yes**, {device_name_str} is in rack {actual_rack}."
            else:
                cleaned_result["yes_no_answer"] = f"\u274c **No**, {device_name_str} is NOT in rack {expected_rack}. It is in rack {actual_rack}."

        return cleaned_result
