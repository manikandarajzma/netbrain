"""
Panorama domain module -- MCP tools and helpers for Palo Alto Panorama.

Provides two exported helper functions used by netbrain_tools.py:
    - _add_panorama_zones_to_hops
    - _add_panorama_device_groups_to_hops

And two MCP tool functions:
    - query_panorama_ip_object_group
    - query_panorama_address_group_members
"""

import ssl
import json
import asyncio
import re
import aiohttp
import xml.etree.ElementTree as ET
import urllib.parse
import ipaddress
from typing import Optional, Dict, Any, List

from tools.shared import mcp, _get_llm, ChatPromptTemplate, setup_logging
import panoramaauth

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Helper functions (exported for use by netbrain_tools.py)
# ---------------------------------------------------------------------------

async def _add_panorama_zones_to_hops(simplified_hops: List[Dict[str, Any]]) -> None:
    """
    Helper function to query Panorama for security zones and add them to firewall hops.

    Args:
        simplified_hops: List of hop dictionaries to update with zone information
    """
    # Collect firewall interfaces to query
    firewall_interface_map = {}
    for hop_info in simplified_hops:
        if hop_info.get("is_firewall"):
            fw_name = hop_info.get("firewall_device")
            if fw_name:
                if fw_name not in firewall_interface_map:
                    firewall_interface_map[fw_name] = {"interfaces": [], "hops": []}
                in_intf = hop_info.get("in_interface")
                out_intf = hop_info.get("out_interface")

                # Extract interface names (handle dict structures)
                in_intf_name = None
                out_intf_name = None

                if in_intf:
                    if isinstance(in_intf, dict):
                        in_intf_name = in_intf.get("intfDisplaySchemaObj", {}).get("value") or in_intf.get("PhysicalInftName") or in_intf.get("name")
                    else:
                        in_intf_name = str(in_intf)

                if out_intf:
                    if isinstance(out_intf, dict):
                        out_intf_name = out_intf.get("intfDisplaySchemaObj", {}).get("value") or out_intf.get("PhysicalInftName") or out_intf.get("name")
                    else:
                        out_intf_name = str(out_intf)

                if in_intf_name and in_intf_name not in firewall_interface_map[fw_name]["interfaces"]:
                    firewall_interface_map[fw_name]["interfaces"].append(in_intf_name)
                if out_intf_name and out_intf_name not in firewall_interface_map[fw_name]["interfaces"]:
                    firewall_interface_map[fw_name]["interfaces"].append(out_intf_name)

                logger.debug(f"Server - Collected interfaces for {fw_name}: {firewall_interface_map[fw_name]['interfaces']}")

                firewall_interface_map[fw_name]["hops"].append(hop_info)

    # Query Panorama for zones
    for fw_name, fw_data in firewall_interface_map.items():
        if fw_data["interfaces"]:
            try:
                zones = await panoramaauth.get_zones_for_firewall_interfaces(
                    firewall_name=fw_name,
                    interfaces=fw_data["interfaces"],
                    template="Global"  # Explicitly use "Global" template
                )

                # Add zone information to firewall hops
                logger.debug(f"Server - Adding zones to {len(fw_data['hops'])} hops for {fw_name}")
                for hop_info in fw_data["hops"]:
                    in_intf = hop_info.get("in_interface")
                    out_intf = hop_info.get("out_interface")

                    # Extract interface names again for matching
                    in_intf_name = None
                    out_intf_name = None

                    if in_intf:
                        if isinstance(in_intf, dict):
                            in_intf_name = in_intf.get("intfDisplaySchemaObj", {}).get("value") or in_intf.get("PhysicalInftName") or in_intf.get("name")
                        else:
                            in_intf_name = str(in_intf)

                    if out_intf:
                        if isinstance(out_intf, dict):
                            out_intf_name = out_intf.get("intfDisplaySchemaObj", {}).get("value") or out_intf.get("PhysicalInftName") or out_intf.get("name")
                        else:
                            out_intf_name = str(out_intf)

                    logger.debug(f"Server - Matching zones for {fw_name}: in_intf_name={in_intf_name}, out_intf_name={out_intf_name}, zones={zones}")

                    # Match zones with case-insensitive interface name matching
                    if in_intf_name:
                        # Try exact match first
                        if in_intf_name in zones and zones[in_intf_name]:
                            hop_info["in_zone"] = zones[in_intf_name]
                            logger.debug(f"Server - Set in_zone for {fw_name} hop to {zones[in_intf_name]} (exact match)")
                        else:
                            # Try case-insensitive match
                            in_intf_lower = in_intf_name.lower()
                            matched_zone = None
                            for zone_intf, zone_name in zones.items():
                                if zone_intf and zone_intf.lower() == in_intf_lower:
                                    matched_zone = zone_name
                                    break

                            if matched_zone:
                                hop_info["in_zone"] = matched_zone
                                logger.debug(f"Server - Set in_zone for {fw_name} hop to {matched_zone} (case-insensitive match)")

                    if out_intf_name:
                        # Try exact match first
                        if out_intf_name in zones and zones[out_intf_name]:
                            hop_info["out_zone"] = zones[out_intf_name]
                            logger.debug(f"Server - Set out_zone for {fw_name} hop to {zones[out_intf_name]} (exact match)")
                        else:
                            # Try case-insensitive match
                            out_intf_lower = out_intf_name.lower()
                            matched_zone = None
                            for zone_intf, zone_name in zones.items():
                                if zone_intf and zone_intf.lower() == out_intf_lower:
                                    matched_zone = zone_name
                                    break

                            if matched_zone:
                                hop_info["out_zone"] = matched_zone
                                logger.debug(f"Server - Set out_zone for {fw_name} hop to {matched_zone} (case-insensitive match)")

                logger.debug(f"Zones for {fw_name}: {zones}")
            except Exception as e:
                logger.debug(f"Error querying Panorama for {fw_name}: {str(e)}")
                import traceback
                logger.debug(f"Panorama query traceback: {traceback.format_exc()}")


async def _add_panorama_device_groups_to_hops(simplified_hops: List[Dict[str, Any]]) -> None:
    """
    Helper function to query Panorama for device groups and add them to firewall hops.

    Args:
        simplified_hops: List of hop dictionaries to update with device group information
    """
    # Collect unique firewall names
    firewall_names = set()
    firewall_hop_map = {}

    for hop_info in simplified_hops:
        if hop_info.get("is_firewall"):
            fw_name = hop_info.get("firewall_device")
            if fw_name:
                firewall_names.add(fw_name)
                if fw_name not in firewall_hop_map:
                    firewall_hop_map[fw_name] = []
                firewall_hop_map[fw_name].append(hop_info)

    if not firewall_names:
        logger.debug(f"Server - No firewalls found in hops for device group query")
        return

    # Query Panorama for device groups
    try:
        firewall_list = list(firewall_names)
        logger.debug(f"Server - Querying device groups for firewalls: {firewall_list}")

        device_groups = await panoramaauth.get_device_groups_for_firewalls(
            firewall_names=firewall_list
        )

        logger.debug(f"Server - Device groups returned: {device_groups}")

        # Add device group information to firewall hops
        for fw_name, hops in firewall_hop_map.items():
            device_group = device_groups.get(fw_name)
            if device_group:
                logger.debug(f"Server - Adding device group '{device_group}' to {len(hops)} hops for {fw_name}")
                for hop_info in hops:
                    hop_info["device_group"] = device_group
                    logger.debug(f"Server - Set device_group for {fw_name} hop to {device_group}")
            else:
                logger.debug(f"Server - No device group found for {fw_name}")

    except Exception as e:
        logger.debug(f"Error querying Panorama device groups: {str(e)}")
        import traceback
        logger.debug(f"Panorama device group query traceback: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# MCP tool functions
# ---------------------------------------------------------------------------

@mcp.tool()
async def query_panorama_ip_object_group(
    ip_address: str,
    device_group: Optional[str] = None,
    vsys: str = "vsys1"
) -> Dict[str, Any]:
    """
    query_panorama_ip_object_group: Use ONLY when user has ONE IP and asks which address group contains it (e.g. "what group is 10.0.0.1 in"). NEVER use for "path allowed", "traffic allowed", or "check if traffic from X to Y" — use check_path_allowed for those. Input: ip_address (one only). Output: list of groups.

    **CRITICAL DISTINCTION - DO NOT CONFUSE WITH query_panorama_address_group_members:**
    - **This tool (query_panorama_ip_object_group):** Query has an IP address → finds which groups contain that IP
      - Example: "what address group is 11.0.0.0/24 part of" → INPUT: IP "11.0.0.0/24", OUTPUT: groups
      - Example: "which group contains 11.0.0.1" → INPUT: IP "11.0.0.1", OUTPUT: groups
    - **Other tool (query_panorama_address_group_members):** Query has a group name → lists IPs in that group
      - Example: "what IPs are in address group leander_web" → INPUT: group "leander_web", OUTPUT: IPs
      - **DO NOT use this tool (query_panorama_ip_object_group) for queries that have a group name and ask for IPs**

    **QUICK CHECK: Does the user query contain a DOT (.)?**
    - YES (e.g., "11.0.0.1", "192.168.1.1") → This is an IP address → You can use this tool OR ask clarification if it's just the IP with no context
    - NO (e.g., "leander-dc-leaf6") → This is NOT an IP address → DO NOT use this tool → Use get_device_rack_location instead

    **ABSOLUTE RULE #1: If the user query is JUST an IP address (like "11.0.0.1", "11.0.0.2", "192.168.1.1") with NO other words, you MUST ask for clarification first. DO NOT immediately select this tool.**

    **ABSOLUTE RULE #2: IP addresses have DOTS (.) - Device names have DASHES (-). They are completely different.**
    - "11.0.0.1" has dots → it's an IP address → you can use this tool OR ask clarification
    - "leander-dc-leaf6" has dashes → it's a device name → DO NOT use this tool → use get_device_rack_location instead

    **ABSOLUTE RULE #3: When the query is JUST an IP address (no context), ask: "What would you like to do with [IP]? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path"**

    **CRITICAL: This tool is for IP ADDRESSES, NOT for device names.**

    **CRITICAL: When to use this tool:**
    - Use this tool when the query contains an IP ADDRESS (a string with DOTS like "11.0.0.1", "11.0.0.2", "192.168.1.100", "11.0.0.0/24")
    - **If the query is JUST an IP address (like "11.0.0.1" or "11.0.0.2") without explicit context, ask for clarification first using the standard format**
    - **CRITICAL DISTINCTION:**
      - Query asks "what address group is [IP] part of" or "which address group contains [IP]" or "what object group is [IP] in" → This tool (query_panorama_ip_object_group) - you have an IP and want to find which groups contain it
      - Query asks "what IPs are in address group [NAME]" or "list IPs in group [NAME]" → DO NOT use this tool → use query_panorama_address_group_members instead - you have a group name and want to list its members
    - Use this tool when the query explicitly asks about "address group", "object group", "what group", "which group" FOR an IP address (the IP is the input, groups are the output)
    - Use this tool when querying Panorama for IP address membership in address objects or address groups
    - Examples: "what address group is 11.0.0.0/24 part of" → ip_address="11.0.0.0/24" → use this tool
    - Examples: "what address group 10.0.0.254 belongs to" → ip_address="10.0.0.254" → use this tool
    - Examples: "which object group contains 192.168.1.100" → ip_address="192.168.1.100" → use this tool
    - Examples: "find address group for 11.0.0.1" → ip_address="11.0.0.1" → use this tool
    - Examples: "11.0.0.1" (just IP) → ask for clarification: "What would you like to do with 11.0.0.1? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path"
    - Examples: "11.0.0.2" (just IP) → ask for clarification: "What would you like to do with 11.0.0.2? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path"
    - This tool queries Panorama (firewall management), NOT NetBox (rack/device inventory)
    - **CRITICAL: IP addresses have DOTS (.), device names have DASHES (-) - they are completely different. If you see dots, it's an IP address → use this tool or ask clarification. If you see dashes, it's a device name → use get_device_rack_location.**

    **HANDLING FOLLOW-UP RESPONSES:**
    - If conversation history shows a previous clarification question was asked in the standard format: "What would you like to do with [IP]? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path"
    - AND the current query is just "1" or "one" → this means the user selected option 1 (Query Panorama for object groups)
    - **CRITICAL: You MUST use this tool (query_panorama_ip_object_group) when user responds "1" to a clarification question that lists "1) Query Panorama for object groups"**
    - **CRITICAL: The standard clarification question order is: 1) Panorama, 2) Device, 3) Rack, 4) Network Path - if you see "1" and the question lists "1) Query Panorama for object groups", use this tool**
    - Extract the IP address from EARLIER messages in the conversation history (the message immediately before the clarification question)
    - Use this tool (query_panorama_ip_object_group) with the IP address from history
    - Example: User says "11.0.0.1" → clarification asked with "1) Query Panorama for object groups, 2) Look up device..." → user responds "1" → use this tool (query_panorama_ip_object_group) with ip_address="11.0.0.1" (from history)
    - **DO NOT use get_device_rack_location when user responds "1" to a clarification question - "1" ALWAYS means Panorama query in the standard format**

    **IMPORTANT: Do NOT confuse with other tools:**
    - This is NOT for rack queries (use get_rack_details for rack names like "A4")
    - This is NOT for device queries (use get_device_rack_location for device names with dashes like "leander-dc-leaf6")
    - This tool does NOT use "site" parameter - Panorama uses "device_group" (firewall device groups), NOT NetBox sites
    - If the query is JUST an IP address without context (like "11.0.0.1"), ask for clarification mentioning ALL possible intents
    - **CRITICAL: When generating clarification questions, ALWAYS use this EXACT order:**
      * "What would you like to do with [IP]? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path"
      * This order MUST be consistent: Panorama is ALWAYS option 1, device lookup is ALWAYS option 2, rack lookup is ALWAYS option 3, network path is ALWAYS option 4
      * DO NOT change the order - it must always be: Panorama (1), Device (2), Rack (3), Network Path (4)
    - When generating clarification questions for ambiguous IP addresses:
      * ALWAYS include Panorama as option 1
      * Ask what the user wants to DO with the IP (query Panorama, look up device, look up rack)
      * DO NOT ask for "site" when the intent is about Panorama/object groups - Panorama doesn't use sites
    - This is for Panorama address/object group queries for IP addresses

    **Query variations (all → query_panorama_ip_object_group; input is an IP/CIDR with dots; do NOT use for device names with dashes → use get_device_rack_location):**
    - "what address group is 11.0.0.0/24 part of?" / "which group contains 11.0.0.1?"
    - "what object group is 10.0.0.254 in?" / "find address group for 192.168.1.100"
    - "which address group has 11.0.0.1?" / "what group does 10.0.0.1 belong to?"
    - "panorama what group is 11.0.0.1 in?" / "in panorama which group contains 11.0.0.1?"
    - If query has a name with DASHES (e.g. leander-dc-border-leaf1) or says "netbox" / "where is" for a device → use get_device_rack_location, NOT this tool.

    This tool searches Panorama for address objects and address groups containing the specified IP address.
    It checks both shared objects and device-group specific objects.
    Additionally, it queries for security and NAT policies that use the found address groups.

    Args:
        ip_address: IP address to search for (e.g., "192.168.1.100", "10.0.0.1", "10.0.0.254")
        device_group: Optional device group name to search within (if None, searches shared objects)
        vsys: VSYS name (default: "vsys1")

    Returns:
        dict: Object group information including:
            - ip_address: The queried IP address
            - address_objects: List of address objects containing this IP
            - address_groups: List of address groups containing this IP or its address objects
            - policies: List of security and NAT policies that use the found address groups
            - device_group: Device group where objects were found (if applicable)
            - error: Error message if query fails

    **Examples:**
    - Query: "what address group 10.0.0.254 belongs to" → ip_address="10.0.0.254", device_group=None
    - Query: "which object group contains 192.168.1.100" → ip_address="192.168.1.100", device_group=None
    - Query: "find group for IP 10.0.0.1" → ip_address="10.0.0.1", device_group=None
    """
    import xml.etree.ElementTree as ET
    import urllib.parse
    import ipaddress

    logger.debug(f"query_panorama_ip_object_group called with ip_address={ip_address}, device_group={device_group}, vsys={vsys}")

    # Validate IP address or CIDR notation
    query_ip = None
    query_network = None
    is_cidr = '/' in ip_address

    try:
        if is_cidr:
            # CIDR notation - validate as network
            # Use strict=False to allow host bits, but normalize it
            query_network = ipaddress.ip_network(ip_address, strict=False)
            query_ip = query_network.network_address  # Use network address for matching
            logger.debug(f"Query is CIDR: {ip_address} -> normalized network: {query_network}")
        else:
            # Single IP address
            query_ip = ipaddress.ip_address(ip_address)
            logger.debug(f"Query is single IP: {ip_address}")
    except (ValueError, ipaddress.AddressValueError) as e:
        return {
            "ip_address": ip_address,
            "error": f"Invalid IP address or CIDR format: {ip_address} - {str(e)}"
        }

    # Get API key from panoramaauth
    api_key = await panoramaauth.get_api_key()
    if not api_key:
        return {
            "ip_address": ip_address,
            "error": "Failed to authenticate with Panorama. Check credentials in panoramaauth.py"
        }

    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    result = {
        "ip_address": ip_address,
        "address_objects": [],
        "address_groups": [],
        "device_group": device_group,
        "vsys": vsys
    }

    # Get Panorama URL from panoramaauth
    panorama_url = panoramaauth.PANORAMA_URL

    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Query address objects to find ones containing this IP
            # Build list of locations to search
            locations = []

            # If device_group is specified, only search that device group
            if device_group:
                locations.append(("device-group", device_group))
            else:
                # If no device_group specified, search shared AND all device groups
                locations.append(("shared", None))

                # Get list of all device groups to search
                logger.debug(f"Starting device group discovery (device_group=None, will search all groups)")
                try:
                    # Device groups are under /config/devices/entry[@name='localhost.localdomain']/device-group/entry
                    # First, try to get the device groups from the correct location
                    dg_list_url = f"{panorama_url}/api/?type=config&action=get&xpath=/config/devices/entry[@name='localhost.localdomain']/device-group/entry&key={api_key}"
                    logger.debug(f"Querying device groups list from: {dg_list_url[:200]}...")
                    async with session.get(dg_list_url, ssl=ssl_context, timeout=15) as dg_response:
                        logger.debug(f"Device groups list response status: {dg_response.status}")
                        if dg_response.status == 200:
                            dg_xml = await dg_response.text()
                            logger.debug(f"Device groups XML response length: {len(dg_xml)}")
                            logger.debug(f"Device groups XML (first 500 chars): {dg_xml[:500]}")
                            try:
                                dg_root = ET.fromstring(dg_xml)
                                dg_entries = dg_root.findall('.//entry')
                                logger.debug(f"Found {len(dg_entries)} device group entries in XML")
                                for dg_entry in dg_entries:
                                    dg_name = dg_entry.get('name')
                                    if dg_name:
                                        locations.append(("device-group", dg_name))
                                        logger.debug(f"Added device group '{dg_name}' to search locations")
                                    else:
                                        logger.debug(f"Device group entry found but no 'name' attribute")
                                logger.debug(f"Total locations to search after device group discovery: {len(locations)}")
                            except ET.ParseError as e:
                                logger.debug(f"Error parsing device groups list XML: {e}")
                                import traceback
                                logger.debug(f"Parse error traceback: {traceback.format_exc()}")
                        else:
                            error_text = await dg_response.text()
                            logger.debug(f"Failed to get device groups list, status: {dg_response.status}, response: {error_text[:500]}")
                except Exception as e:
                    logger.debug(f"Error getting device groups list: {str(e)}")
                    import traceback
                    logger.debug(f"Device group discovery exception traceback: {traceback.format_exc()}")
                    # Continue with just shared if we can't get device groups

            matching_address_objects = []

            for location_type, location_name in locations:
                try:
                    # Build XPath for address objects
                    if location_type == "device-group":
                        xpath = f"/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='{location_name}']/address"
                    else:  # shared
                        xpath = "/config/shared/address"

                    url = f"{panorama_url}/api/?type=config&action=get&xpath={urllib.parse.quote(xpath)}&key={api_key}"
                    logger.debug(f"Querying address objects from {location_type}: {url[:200]}...")

                    async with session.get(url, ssl=ssl_context, timeout=30) as response:
                        if response.status == 200:
                            xml_text = await response.text()
                            logger.debug(f"Address objects XML response length: {len(xml_text)}")

                            try:
                                root = ET.fromstring(xml_text)
                                # Find all address entries
                                entries = root.findall('.//entry')
                                logger.debug(f"Found {len(entries)} address objects in {location_type} {location_name or 'shared'}")

                                for entry in entries:
                                    obj_name = entry.get('name')
                                    if not obj_name:
                                        continue

                                    logger.debug(f"Checking address object '{obj_name}' in {location_type} {location_name or 'shared'}")

                                    # Check different IP formats in the address object
                                    # Check for ip-netmask, ip-range, fqdn, etc.
                                    ip_netmask = entry.find('ip-netmask')
                                    ip_range = entry.find('ip-range')
                                    fqdn = entry.find('fqdn')

                                    matches = False
                                    obj_type = None
                                    obj_value = None

                                    # Debug: print what we found in the entry
                                    if ip_netmask is not None:
                                        logger.debug(f"Object '{obj_name}' has ip-netmask: {ip_netmask.text}")
                                    if ip_range is not None:
                                        logger.debug(f"Object '{obj_name}' has ip-range: {ip_range.text}")
                                    if fqdn is not None:
                                        logger.debug(f"Object '{obj_name}' has fqdn: {fqdn.text}")

                                    if ip_netmask is not None and ip_netmask.text:
                                        # Check if IP matches the netmask/CIDR
                                        obj_value = ip_netmask.text.strip()
                                        obj_type = "ip-netmask"
                                        try:
                                            if '/' in obj_value:
                                                # CIDR notation in object - check if query IP/network overlaps
                                                obj_network = ipaddress.ip_network(obj_value, strict=False)
                                                logger.debug(f"Comparing query {ip_address} (is_cidr={is_cidr}) with object {obj_name} value {obj_value}")
                                                if is_cidr:
                                                    # Both are CIDR - check if networks are the same
                                                    # For exact match, compare the normalized networks
                                                    # ip_network() automatically normalizes, so direct comparison should work
                                                    matches = (query_network == obj_network)
                                                    if not matches:
                                                        # Also check if they overlap (one contains the other)
                                                        matches = query_network.overlaps(obj_network)
                                                    logger.debug(f"CIDR vs CIDR: query_net={query_network} (normalized), obj_net={obj_network} (normalized), exact_match={query_network == obj_network}, overlaps={query_network.overlaps(obj_network) if query_network != obj_network else False}, final_matches={matches}")
                                                else:
                                                    # Query is single IP, object is CIDR - check if IP is in network
                                                    matches = query_ip in obj_network
                                                    logger.debug(f"Single IP in CIDR: query_ip={query_ip}, obj_net={obj_network}, matches={matches}")
                                            else:
                                                # Single IP in object
                                                obj_ip = ipaddress.ip_address(obj_value)
                                                if is_cidr:
                                                    # Query is CIDR, object is single IP - check if IP is in query network
                                                    matches = obj_ip in query_network
                                                    logger.debug(f"CIDR contains IP: query_net={query_network}, obj_ip={obj_ip}, matches={matches}")
                                                else:
                                                    # Both are single IPs - compare directly
                                                    matches = (query_ip == obj_ip)
                                                    logger.debug(f"IP vs IP: query_ip={query_ip}, obj_ip={obj_ip}, matches={matches}")
                                        except (ValueError, ipaddress.AddressValueError) as e:
                                            matches = False
                                            logger.debug(f"Error comparing {ip_address} with {obj_value}: {e}")

                                    elif ip_range is not None and ip_range.text:
                                        obj_value = ip_range.text
                                        obj_type = "ip-range"
                                        # Check if IP is in range (format: "start-end")
                                        if '-' in obj_value:
                                            try:
                                                start_ip, end_ip = obj_value.split('-', 1)
                                                start = ipaddress.ip_address(start_ip.strip())
                                                end = ipaddress.ip_address(end_ip.strip())
                                                if is_cidr:
                                                    # Query is CIDR - check if any IP in the network is in range
                                                    # Simple check: if network address is in range
                                                    matches = (start <= query_ip <= end)
                                                else:
                                                    # Query is single IP - check if it's between start and end
                                                    matches = (start <= query_ip <= end)
                                            except (ValueError, ipaddress.AddressValueError):
                                                matches = False
                                        else:
                                            matches = False

                                    elif fqdn is not None and fqdn.text:
                                        obj_type = "fqdn"
                                        obj_value = fqdn.text
                                        # FQDN doesn't match IP directly
                                        matches = False

                                    if matches:
                                        matching_address_objects.append({
                                            "name": obj_name,
                                            "type": obj_type,
                                            "value": obj_value,
                                            "location": location_type,
                                            "device_group": location_name if location_type == "device-group" else None
                                        })
                                        logger.debug(f"\u2713 MATCH FOUND! Address object: {obj_name} ({obj_type}: {obj_value}) in {location_type} {location_name or 'shared'}")
                                    else:
                                        logger.debug(f"\u2717 No match for object '{obj_name}' (value: {obj_value or 'N/A'})")

                            except ET.ParseError as e:
                                logger.debug(f"Error parsing address objects XML: {e}")
                        else:
                            logger.debug(f"Address objects query failed with status {response.status}")

                except Exception as e:
                    logger.debug(f"Error querying address objects from {location_type}: {str(e)}")

            result["address_objects"] = matching_address_objects

            logger.debug(f"Finished searching address objects. Found {len(matching_address_objects)} matching objects.")

            # Step 2: Query address groups to find ones containing the matching address objects
            for location_type, location_name in locations:
                try:
                    # Build XPath for address groups
                    if location_type == "device-group":
                        xpath = f"/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='{location_name}']/address-group"
                    else:  # shared
                        xpath = "/config/shared/address-group"

                    url = f"{panorama_url}/api/?type=config&action=get&xpath={urllib.parse.quote(xpath)}&key={api_key}"
                    logger.debug(f"Querying address groups from {location_type}: {url[:200]}...")

                    async with session.get(url, ssl=ssl_context, timeout=30) as response:
                        if response.status == 200:
                            xml_text = await response.text()

                            try:
                                root = ET.fromstring(xml_text)
                                entries = root.findall('.//entry')

                                for entry in entries:
                                    group_name = entry.get('name')
                                    if not group_name:
                                        continue

                                    # Get static members (address objects in the group)
                                    static = entry.find('static')
                                    if static is not None:
                                        members = static.findall('member')
                                        member_names = [m.text for m in members if m.text]

                                        # Check if any matching address object is in this group
                                        for addr_obj in matching_address_objects:
                                            if addr_obj["name"] in member_names:
                                                # Found a matching group - now get all members with their IP values
                                                group_members = []

                                                # Query each member address object to get its IP value
                                                for member_name in member_names:
                                                    try:
                                                        # Build XPath for the address object
                                                        if location_type == "device-group":
                                                            obj_xpath = f"/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='{location_name}']/address/entry[@name='{member_name}']"
                                                        else:  # shared
                                                            obj_xpath = f"/config/shared/address/entry[@name='{member_name}']"

                                                        obj_url = f"{panorama_url}/api/?type=config&action=get&xpath={urllib.parse.quote(obj_xpath)}&key={api_key}"
                                                        logger.debug(f"Querying address object '{member_name}' from {location_type} for group '{group_name}': {obj_url[:200]}...")

                                                        async with session.get(obj_url, ssl=ssl_context, timeout=30) as obj_response:
                                                            if obj_response.status == 200:
                                                                obj_xml = await obj_response.text()

                                                                try:
                                                                    obj_root = ET.fromstring(obj_xml)
                                                                    obj_entry = obj_root.find('.//entry')

                                                                    if obj_entry is not None:
                                                                        # Extract IP value from different possible formats
                                                                        ip_netmask = obj_entry.find('ip-netmask')
                                                                        ip_range = obj_entry.find('ip-range')
                                                                        fqdn = obj_entry.find('fqdn')

                                                                        obj_type = None
                                                                        obj_value = None

                                                                        if ip_netmask is not None and ip_netmask.text:
                                                                            obj_type = "ip-netmask"
                                                                            obj_value = ip_netmask.text.strip()
                                                                        elif ip_range is not None and ip_range.text:
                                                                            obj_type = "ip-range"
                                                                            obj_value = ip_range.text.strip()
                                                                        elif fqdn is not None and fqdn.text:
                                                                            obj_type = "fqdn"
                                                                            obj_value = fqdn.text.strip()

                                                                        group_members.append({
                                                                            "name": member_name,
                                                                            "type": obj_type,
                                                                            "value": obj_value,
                                                                            "location": location_type,
                                                                            "device_group": location_name if location_type == "device-group" else None
                                                                        })
                                                                        logger.debug(f"\u2713 Found address object '{member_name}' in group '{group_name}': {obj_type}={obj_value}")

                                                                except ET.ParseError as e:
                                                                    logger.debug(f"Error parsing address object '{member_name}' XML: {e}")
                                                                    # Still add the member name even if we can't get the value
                                                                    group_members.append({
                                                                        "name": member_name,
                                                                        "type": "unknown",
                                                                        "value": None,
                                                                        "location": location_type,
                                                                        "device_group": location_name if location_type == "device-group" else None
                                                                    })
                                                            else:
                                                                logger.debug(f"Address object '{member_name}' query failed with status {obj_response.status}")
                                                                # Still add the member name even if we can't get the value
                                                                group_members.append({
                                                                    "name": member_name,
                                                                    "type": "unknown",
                                                                    "value": None,
                                                                    "location": location_type,
                                                                    "device_group": location_name if location_type == "device-group" else None
                                                                })

                                                    except Exception as e:
                                                        logger.debug(f"Error querying address object '{member_name}': {str(e)}")
                                                        # Still add the member name even if we can't get the value
                                                        group_members.append({
                                                            "name": member_name,
                                                            "type": "unknown",
                                                            "value": None,
                                                            "location": location_type,
                                                            "device_group": location_name if location_type == "device-group" else None
                                                        })

                                                result["address_groups"].append({
                                                    "name": group_name,
                                                    "location": location_type,
                                                    "device_group": location_name if location_type == "device-group" else None,
                                                    "contains_address_object": addr_obj["name"],
                                                    "members": group_members  # Include all members with their IP values
                                                })
                                                logger.debug(f"Found address group '{group_name}' containing address object '{addr_obj['name']}' with {len(group_members)} total members")
                                                break

                            except ET.ParseError as e:
                                logger.debug(f"Error parsing address groups XML: {e}")
                        else:
                            logger.debug(f"Address groups query failed with status {response.status}")

                except Exception as e:
                    logger.debug(f"Error querying address groups from {location_type}: {str(e)}")

            # Step 3: Query policies (security and NAT) that use the found address groups AND address objects
            result["policies"] = []
            policies_by_group = {}  # Track policies per address group/object

            # Collect address objects and their locations
            addr_object_info = {}
            locations_to_query = set()
            for addr_obj in matching_address_objects:
                obj_name = addr_obj["name"]
                location_type = addr_obj["location"]
                location_name = addr_obj.get("device_group")
                addr_object_info[obj_name] = {
                    "location": location_type,
                    "device_group": location_name
                }
                locations_to_query.add((location_type, location_name))

            # Collect address groups and their locations
            group_info = {}
            if result["address_groups"]:
                logger.debug(f"Querying policies for {len(result['address_groups'])} address groups and {len(matching_address_objects)} address objects")

                for addr_group in result["address_groups"]:
                    group_name = addr_group["name"]
                    location_type = addr_group["location"]
                    location_name = addr_group.get("device_group")
                    group_info[group_name] = {
                        "location": location_type,
                        "device_group": location_name
                    }
                    locations_to_query.add((location_type, location_name))

            # Also query policies if we have address objects (even without groups)
            if matching_address_objects and not result["address_groups"]:
                logger.debug(f"Querying policies for {len(matching_address_objects)} address objects (no address groups found)")

            # Query policies if we have either groups or objects
            if locations_to_query:
                logger.debug(f"Will query policies from {len(locations_to_query)} location(s): {list(locations_to_query)}")
                logger.debug(f"Looking for policies using groups: {list(group_info.keys())}, objects: {list(addr_object_info.keys())}")

                # Query policies from the same locations where groups/objects were found
                for location_type, location_name in locations_to_query:
                    try:
                        # Query Security Policies - both Pre and Post Rules
                        security_rulebases = ["pre-rulebase", "post-rulebase"]

                        for rulebase in security_rulebases:
                            if location_type == "device-group":
                                sec_xpath = f"/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='{location_name}']/{rulebase}/security/rules/entry"
                            else:  # shared
                                sec_xpath = f"/config/shared/{rulebase}/security/rules/entry"

                            sec_url = f"{panorama_url}/api/?type=config&action=get&xpath={urllib.parse.quote(sec_xpath)}&key={api_key}"
                            logger.debug(f"Querying security policies from {location_type} {rulebase}: {sec_url[:200]}...")

                            async with session.get(sec_url, ssl=ssl_context, timeout=30) as sec_response:
                                if sec_response.status == 200:
                                    sec_xml = await sec_response.text()
                                    logger.debug(f"Security policies XML response length: {len(sec_xml)} chars")
                                    try:
                                        sec_root = ET.fromstring(sec_xml)
                                        sec_entries = sec_root.findall('.//entry')
                                        logger.debug(f"Found {len(sec_entries)} security policy entries in {rulebase} for {location_type} {location_name or 'shared'}")

                                        for entry in sec_entries:
                                            rule_name = entry.get('name')
                                            if not rule_name:
                                                continue

                                            # Check source and destination for address group references
                                            source = entry.find('source')
                                            destination = entry.find('destination')

                                            # Get source and destination members for checking
                                            source_members_list = source.findall('member') if source is not None else []
                                            dest_members_list = destination.findall('member') if destination is not None else []
                                            source_members = [m.text for m in source_members_list if m.text]
                                            dest_members = [m.text for m in dest_members_list if m.text]

                                            # Debug: log policy details
                                            if rule_name in ["ai-test"] or any(g in source_members + dest_members for g in group_info.keys()) or any(o in source_members + dest_members for o in addr_object_info.keys()):
                                                logger.debug(f"Checking policy '{rule_name}' - source: {source_members}, dest: {dest_members}")

                                            # Check if any of our address groups are referenced
                                            matched_groups = []
                                            for group_name in group_info.keys():
                                                source_has_group = any(m == group_name for m in source_members)
                                                dest_has_group = any(m == group_name for m in dest_members)

                                                if source_has_group or dest_has_group:
                                                    matched_groups.append(group_name)

                                            # Check if any of our address objects are referenced directly
                                            matched_objects = []
                                            for obj_name in addr_object_info.keys():
                                                source_has_obj = any(m == obj_name for m in source_members)
                                                dest_has_obj = any(m == obj_name for m in dest_members)

                                                if source_has_obj or dest_has_obj:
                                                    matched_objects.append(obj_name)

                                            # If we found any matches (groups or objects), add the policy
                                            if matched_groups or matched_objects:
                                                # Get action
                                                action_elem = entry.find('action')
                                                action = action_elem.text if action_elem is not None else "unknown"

                                                # Get service
                                                service_elem = entry.find('service')
                                                services = [s.text for s in service_elem.findall('member')] if service_elem is not None else []

                                                policy_key = f"{location_type}:{location_name or 'shared'}:{rule_name}:{rulebase}"
                                                if policy_key not in policies_by_group:
                                                    policies_by_group[policy_key] = {
                                                        "name": rule_name,
                                                        "type": "security",
                                                        "rulebase": rulebase,  # Track if it's pre or post
                                                        "location": location_type,
                                                        "device_group": location_name if location_type == "device-group" else None,
                                                        "action": action,
                                                        "source": source_members,
                                                        "destination": dest_members,
                                                        "services": services,
                                                        "address_groups": [],
                                                        "address_objects": []
                                                    }

                                                # Add matched groups
                                                for group_name in matched_groups:
                                                    if group_name not in policies_by_group[policy_key]["address_groups"]:
                                                        policies_by_group[policy_key]["address_groups"].append(group_name)

                                                # Add matched objects
                                                for obj_name in matched_objects:
                                                    if obj_name not in policies_by_group[policy_key]["address_objects"]:
                                                        policies_by_group[policy_key]["address_objects"].append(obj_name)

                                                match_desc = []
                                                if matched_groups:
                                                    match_desc.append(f"address groups: {', '.join(matched_groups)}")
                                                if matched_objects:
                                                    match_desc.append(f"address objects: {', '.join(matched_objects)}")

                                                logger.debug(f"Found security policy '{rule_name}' ({rulebase}) using {', '.join(match_desc)} in {location_type} {location_name or 'shared'}")

                                    except ET.ParseError as e:
                                        logger.debug(f"Error parsing security policies XML from {rulebase}: {e}")
                                elif sec_response.status == 404:
                                    logger.debug(f"No security policies found in {rulebase} for {location_type} {location_name or 'shared'} (404)")
                                else:
                                    logger.debug(f"Security policies query failed with status {sec_response.status} for {rulebase}")

                        # Query NAT Policies - both Pre and Post Rules
                        nat_rulebases = ["pre-rulebase", "post-rulebase"]

                        for rulebase in nat_rulebases:
                            if location_type == "device-group":
                                nat_xpath = f"/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='{location_name}']/{rulebase}/nat/rules/entry"
                            else:  # shared
                                nat_xpath = f"/config/shared/{rulebase}/nat/rules/entry"

                        nat_url = f"{panorama_url}/api/?type=config&action=get&xpath={urllib.parse.quote(nat_xpath)}&key={api_key}"
                        logger.debug(f"Querying NAT policies from {location_type}: {nat_url[:200]}...")

                        async with session.get(nat_url, ssl=ssl_context, timeout=30) as nat_response:
                            if nat_response.status == 200:
                                nat_xml = await nat_response.text()
                                try:
                                    nat_root = ET.fromstring(nat_xml)
                                    nat_entries = nat_root.findall('.//entry')

                                    for entry in nat_entries:
                                        rule_name = entry.get('name')
                                        if not rule_name:
                                            continue

                                        # Check source-translation and destination-translation for address group references
                                        source_translation = entry.find('source-translation')
                                        destination_translation = entry.find('destination-translation')

                                        # Also check source and destination
                                        source = entry.find('source')
                                        destination = entry.find('destination')

                                        # Get source and destination members for checking
                                        source_members_list = source.findall('member') if source is not None else []
                                        dest_members_list = destination.findall('member') if destination is not None else []
                                        source_members = [m.text for m in source_members_list if m.text]
                                        dest_members = [m.text for m in dest_members_list if m.text]

                                        # Check if any of our address groups or objects are referenced
                                        matched_groups = []
                                        matched_objects = []
                                        nat_type = None

                                        # Check source for groups and objects
                                        if source is not None:
                                            for group_name in group_info.keys():
                                                if any(m == group_name for m in source_members):
                                                    matched_groups.append(group_name)
                                                    nat_type = "source" if nat_type is None else f"{nat_type}/source"

                                            for obj_name in addr_object_info.keys():
                                                if any(m == obj_name for m in source_members):
                                                    matched_objects.append(obj_name)
                                                    nat_type = "source" if nat_type is None else f"{nat_type}/source"

                                        # Check destination for groups and objects
                                        if destination is not None:
                                            for group_name in group_info.keys():
                                                if any(m == group_name for m in dest_members):
                                                    if group_name not in matched_groups:
                                                        matched_groups.append(group_name)
                                                    nat_type = "destination" if nat_type is None else f"{nat_type}/destination"

                                            for obj_name in addr_object_info.keys():
                                                if any(m == obj_name for m in dest_members):
                                                    if obj_name not in matched_objects:
                                                        matched_objects.append(obj_name)
                                                    nat_type = "destination" if nat_type is None else f"{nat_type}/destination"

                                        # Check source-translation (for groups only, as objects are typically not in translation)
                                        if source_translation is not None:
                                            static_ip = source_translation.find('static-ip')
                                            if static_ip is not None:
                                                translated_addr = static_ip.find('translated-address')
                                                if translated_addr is not None:
                                                    for group_name in group_info.keys():
                                                        if translated_addr.text == group_name:
                                                            if group_name not in matched_groups:
                                                                matched_groups.append(group_name)
                                                            nat_type = "source-translation" if nat_type is None else f"{nat_type}/source-translation"

                                        # Check destination-translation (for groups only)
                                        if destination_translation is not None:
                                            static_ip = destination_translation.find('static-ip')
                                            if static_ip is not None:
                                                translated_addr = static_ip.find('translated-address')
                                                if translated_addr is not None:
                                                    for group_name in group_info.keys():
                                                        if translated_addr.text == group_name:
                                                            if group_name not in matched_groups:
                                                                matched_groups.append(group_name)
                                                            nat_type = "destination-translation" if nat_type is None else f"{nat_type}/destination-translation"

                                        # If we found any matches (groups or objects), add the policy
                                        if matched_groups or matched_objects:
                                            # Get service
                                            service_elem = entry.find('service')
                                            services = [s.text for s in service_elem.findall('member')] if service_elem is not None else []

                                            policy_key = f"{location_type}:{location_name or 'shared'}:{rule_name}:{rulebase}"
                                            if policy_key not in policies_by_group:
                                                policies_by_group[policy_key] = {
                                                    "name": rule_name,
                                                    "type": "nat",
                                                    "rulebase": rulebase,  # Track if it's pre or post
                                                    "location": location_type,
                                                    "device_group": location_name if location_type == "device-group" else None,
                                                    "nat_type": nat_type,
                                                    "source": source_members,
                                                    "destination": dest_members,
                                                    "services": services,
                                                    "address_groups": [],
                                                    "address_objects": []
                                                }

                                            # Add matched groups
                                            for group_name in matched_groups:
                                                if group_name not in policies_by_group[policy_key]["address_groups"]:
                                                    policies_by_group[policy_key]["address_groups"].append(group_name)

                                            # Add matched objects
                                            for obj_name in matched_objects:
                                                if obj_name not in policies_by_group[policy_key]["address_objects"]:
                                                    policies_by_group[policy_key]["address_objects"].append(obj_name)

                                            match_desc = []
                                            if matched_groups:
                                                match_desc.append(f"address groups: {', '.join(matched_groups)}")
                                            if matched_objects:
                                                match_desc.append(f"address objects: {', '.join(matched_objects)}")

                                            logger.debug(f"Found NAT policy '{rule_name}' ({rulebase}) using {', '.join(match_desc)} in {location_type} {location_name or 'shared'}")

                                except ET.ParseError as e:
                                    logger.debug(f"Error parsing NAT policies XML from {rulebase}: {e}")
                            elif nat_response.status == 404:
                                logger.debug(f"No NAT policies found in {rulebase} for {location_type} {location_name or 'shared'} (404)")
                            else:
                                logger.debug(f"NAT policies query failed with status {nat_response.status} for {rulebase}")

                    except Exception as e:
                        logger.debug(f"Error querying policies from {location_type}: {str(e)}")

                # Convert policies dict to list
                result["policies"] = list(policies_by_group.values())
                logger.debug(f"Found {len(result['policies'])} policies using the address groups/objects")
                if result["policies"]:
                    for policy in result["policies"]:
                        logger.debug(f"Policy '{policy['name']}' ({policy['type']}, {policy.get('rulebase', 'unknown')}) uses groups: {policy.get('address_groups', [])}, objects: {policy.get('address_objects', [])}")
                else:
                    logger.debug(f"No policies found. Searched {len(locations_to_query)} locations. Group info: {list(group_info.keys())}, Object info: {list(addr_object_info.keys())}")

            # If no matches found, provide detailed debug info
            if not matching_address_objects and not result["address_groups"]:
                result["message"] = f"IP address {ip_address} not found in any address objects or address groups"
                result["debug_info"] = {
                    "locations_searched": len(locations),
                    "location_details": [f"{loc_type}: {loc_name or 'shared'}" for loc_type, loc_name in locations]
                }
                locations_str = [f"{loc_type}: {loc_name or 'shared'}" for loc_type, loc_name in locations]
                logger.debug(f"No matches found. Searched {len(locations)} locations: {locations_str}")
            else:
                result["message"] = f"Found {len(matching_address_objects)} address object(s) and {len(result['address_groups'])} address group(s)"
                if result["policies"]:
                    result["message"] += f" and {len(result['policies'])} policy/policies using these groups"
                logger.debug(f"Success! Found {len(matching_address_objects)} address objects, {len(result['address_groups'])} address groups, and {len(result['policies'])} policies")
                # Add top-level "members" so the UI shows "Address group members (IPs)" first (same shape as query_panorama_address_group_members)
                seen = set()
                result["members"] = []
                for ag in result.get("address_groups", []):
                    for m in ag.get("members") or []:
                        key = (m.get("name"), m.get("value"))
                        if key not in seen:
                            seen.add(key)
                            result["members"].append(m)

        # Send result to LLM for analysis and table format summary
        llm = _get_llm()
        if llm is not None and "error" not in result:
            logger.debug(f"LLM available, starting analysis for Panorama IP object group query")
            try:
                from langchain_core.prompts import ChatPromptTemplate

                system_prompt = """You are a network security assistant. The UI will display the data in proper tables. Your job is to write a SHORT narrative summary only (2-4 sentences).

Do NOT use markdown tables, pipe characters, or column layouts. Do NOT output "Table 1", "Table 2", or "--- | ---".

Write a brief summary that:
- STARTS with the direct answer: which address group(s) the IP is in (e.g. "The IP 11.0.0.1 is in address group leander_web.").
- Then mention address object name(s) and how many policies use them, and location (e.g. device-group) if relevant.

Example: "The IP 11.0.0.1 is in address group leander_web. It is in address object test_destination. Three security policies (test, ai-test, ai-test1) use these in device-group leander." Keep it under 2-4 sentences."""

                analysis_prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "Analyze this Panorama query result:\n{query_result}")
                ])

                formatted_messages = analysis_prompt_template.format_messages(
                    query_result=json.dumps(result, indent=2)
                )

                logger.debug(f"Invoking LLM for Panorama analysis")
                response = await asyncio.wait_for(
                    llm.ainvoke(formatted_messages),
                    timeout=30.0
                )
                content = response.content if hasattr(response, 'content') else str(response)

                logger.debug(f"LLM response received: {content[:200]}...")

                # Extract JSON from response if present, otherwise use full content
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
                logger.debug(f"LLM analysis failed: {str(e)}")
                import traceback
                logger.debug(f"LLM analysis traceback: {traceback.format_exc()}")
                # Provide a basic summary if LLM fails
                addr_objects_count = len(result.get("address_objects", []))
                addr_groups_count = len(result.get("address_groups", []))
                if addr_objects_count > 0 or addr_groups_count > 0:
                    result["ai_analysis"] = {
                        "summary": f"IP {ip_address} found in {addr_objects_count} address object(s) and {addr_groups_count} address group(s)."
                    }
                else:
                    result["ai_analysis"] = {
                        "summary": f"IP {ip_address} not found in any address objects or address groups."
                    }

    except Exception as e:
        logger.error(f"Exception querying Panorama for IP object group: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        result["error"] = f"Error querying Panorama: {str(e)}"

    return result


@mcp.tool()
async def query_panorama_address_group_members(
    address_group_name: str,
    device_group: Optional[str] = None,
    vsys: str = "vsys1"
) -> Dict[str, Any]:
    """
    Query Panorama to list all IPs/address objects in a specific address group (input: group name; output: list of members).

    Do NOT use for: "check if traffic from X to Y is allowed", "is path allowed", "is traffic allowed" — use check_path_allowed instead. This tool only lists members of a named address group; it does not check if traffic is allowed between two IPs.

    **CRITICAL DISTINCTION - DO NOT CONFUSE WITH query_panorama_ip_object_group:**
    - **This tool (query_panorama_address_group_members):** Query has a group name → lists IPs in that group
      - Example: "what IPs are in address group leander_web" → INPUT: group "leander_web", OUTPUT: IPs
      - Example: "list IPs in group leander_web" → INPUT: group "leander_web", OUTPUT: IPs
    - **Other tool (query_panorama_ip_object_group):** Query has an IP address → finds which groups contain that IP
      - Example: "what address group is 11.0.0.0/24 part of" → INPUT: IP "11.0.0.0/24", OUTPUT: groups
      - **DO NOT use this tool (query_panorama_address_group_members) for queries that have an IP and ask which groups contain it**

    **CRITICAL: When to use this tool:**
    - Use this tool when the query asks about "what IPs are in address group X", "list IPs in group X", "what addresses are in group X"
    - Use this tool when the query mentions an address group name (e.g., "leander_web", "web_servers", "dmz_hosts")
    - **CRITICAL DISTINCTION:**
      - Query asks "what IPs are in address group [NAME]" or "list IPs in group [NAME]" → This tool (query_panorama_address_group_members) - you have a group name and want to list its members
      - Query asks "what address group is [IP] part of" or "which group contains [IP]" → DO NOT use this tool → use query_panorama_ip_object_group instead - you have an IP and want to find which groups contain it
    - Examples: "what other IPs are in the address group leander_web" → address_group_name="leander_web" → use this tool
    - Examples: "list all IPs in address group web_servers" → address_group_name="web_servers" → use this tool
    - Examples: "what addresses are in the group leander_web" → address_group_name="leander_web" → use this tool
    - This tool queries Panorama (firewall management), NOT NetBox (rack/device inventory)

    **IMPORTANT: Do NOT confuse with other tools:**
    - This is NOT for IP addresses (use query_panorama_ip_object_group to find which groups an IP belongs to)
    - This is NOT for rack queries (use get_rack_details for rack names like "A4")
    - This is NOT for device queries (use get_device_rack_location for device names with dashes like "leander-dc-leaf6")
    - This tool does NOT use "site" parameter - Panorama uses "device_group" (firewall device groups), NOT NetBox sites

    This tool searches Panorama for a specific address group and returns all address objects (and their IP values) that are members of that group.
    It checks both shared address groups and device-group specific address groups.

    Args:
        address_group_name: Name of the address group to query (e.g., "leander_web", "web_servers")
        device_group: Optional device group name to search within (if None, searches shared and all device groups)
        vsys: VSYS name (default: "vsys1")

    Returns:
        dict: Address group members information including:
            - address_group_name: The queried address group name
            - members: List of address objects in the group with their IP values
            - policies: List of security and NAT policies that use this address group
            - device_group: Device group where the group was found (if applicable)
            - location: Location where group was found ("shared" or "device-group")
            - error: Error message if query fails

    **Examples:**
    - Query: "what other IPs are in the address group leander_web" → address_group_name="leander_web", device_group=None
    - Query: "list all IPs in address group web_servers" → address_group_name="web_servers", device_group=None
    - Query: "what addresses are in the group leander_web" → address_group_name="leander_web", device_group=None

    **Query variations (all → query_panorama_address_group_members; input is a GROUP NAME, output is list of IPs):**
    - "what IPs are in address group leander_web?" / "list IPs in group leander_web"
    - "what addresses are in the group web_servers?" / "list members of address group dmz_hosts"
    - "show me IPs in group leander_web" / "members of address group web_servers"
    - "panorama list IPs in group X" / "what's in address group leander_web?"
    - Do NOT use for "which group contains [IP]" → use query_panorama_ip_object_group instead.
    """
    import xml.etree.ElementTree as ET
    import urllib.parse

    logger.debug(f"query_panorama_address_group_members called with address_group_name={address_group_name}, device_group={device_group}, vsys={vsys}")

    # Get API key from panoramaauth
    api_key = await panoramaauth.get_api_key()
    if not api_key:
        return {
            "address_group_name": address_group_name,
            "error": "Failed to authenticate with Panorama. Check credentials in panoramaauth.py"
        }

    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    result = {
        "address_group_name": address_group_name,
        "members": [],
        "device_group": device_group,
        "location": None,
        "vsys": vsys
    }

    # Get Panorama URL from panoramaauth
    panorama_url = panoramaauth.PANORAMA_URL

    try:
        async with aiohttp.ClientSession() as session:
            # Build list of locations to search
            locations = []

            # If device_group is specified, only search that device group
            if device_group:
                locations.append(("device-group", device_group))
            else:
                # If no device_group specified, search shared AND all device groups
                locations.append(("shared", None))

                # Get list of all device groups to search
                logger.debug(f"Starting device group discovery (device_group=None, will search all groups)")
                try:
                    dg_list_url = f"{panorama_url}/api/?type=config&action=get&xpath=/config/devices/entry[@name='localhost.localdomain']/device-group/entry&key={api_key}"
                    logger.debug(f"Querying device groups list from: {dg_list_url[:200]}...")
                    async with session.get(dg_list_url, ssl=ssl_context, timeout=15) as dg_response:
                        if dg_response.status == 200:
                            dg_xml = await dg_response.text()
                            try:
                                dg_root = ET.fromstring(dg_xml)
                                dg_entries = dg_root.findall('.//entry')
                                for dg_entry in dg_entries:
                                    dg_name = dg_entry.get('name')
                                    if dg_name:
                                        locations.append(("device-group", dg_name))
                                        logger.debug(f"Added device group '{dg_name}' to search locations")
                            except ET.ParseError as e:
                                logger.debug(f"Error parsing device groups list XML: {e}")
                except Exception as e:
                    logger.debug(f"Error getting device groups list: {str(e)}")

            # Search for the address group
            found_group = None
            group_location = None
            group_device_group = None

            for location_type, location_name in locations:
                try:
                    # Build XPath for address groups
                    if location_type == "device-group":
                        xpath = f"/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='{location_name}']/address-group/entry[@name='{address_group_name}']"
                    else:  # shared
                        xpath = f"/config/shared/address-group/entry[@name='{address_group_name}']"

                    url = f"{panorama_url}/api/?type=config&action=get&xpath={urllib.parse.quote(xpath)}&key={api_key}"
                    logger.debug(f"Querying address group '{address_group_name}' from {location_type}: {url[:200]}...")

                    async with session.get(url, ssl=ssl_context, timeout=30) as response:
                        if response.status == 200:
                            xml_text = await response.text()
                            logger.debug(f"Address group XML response length: {len(xml_text)}")

                            try:
                                root = ET.fromstring(xml_text)
                                entry = root.find('.//entry')

                                if entry is not None:
                                    found_group = entry
                                    group_location = location_type
                                    group_device_group = location_name if location_type == "device-group" else None
                                    result["location"] = location_type
                                    result["device_group"] = group_device_group
                                    logger.debug(f"\u2713 Found address group '{address_group_name}' in {location_type} {location_name or 'shared'}")
                                    break

                            except ET.ParseError as e:
                                logger.debug(f"Error parsing address group XML: {e}")
                        elif response.status == 404:
                            logger.debug(f"Address group '{address_group_name}' not found in {location_type} {location_name or 'shared'}")
                        else:
                            logger.debug(f"Address group query failed with status {response.status}")

                except Exception as e:
                    logger.debug(f"Error querying address group from {location_type}: {str(e)}")

            if not found_group:
                result["error"] = f"Address group '{address_group_name}' not found in Panorama"
                return result

            # Get static members (address object names) from the group
            static = found_group.find('static')
            if static is not None:
                members = static.findall('member')
                member_names = [m.text for m in members if m.text]
                logger.debug(f"Address group '{address_group_name}' has {len(member_names)} members: {member_names}")

                # Now query each address object to get its IP value
                for member_name in member_names:
                    try:
                        # Build XPath for the address object
                        if group_location == "device-group":
                            obj_xpath = f"/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='{group_device_group}']/address/entry[@name='{member_name}']"
                        else:  # shared
                            obj_xpath = f"/config/shared/address/entry[@name='{member_name}']"

                        obj_url = f"{panorama_url}/api/?type=config&action=get&xpath={urllib.parse.quote(obj_xpath)}&key={api_key}"
                        logger.debug(f"Querying address object '{member_name}' from {group_location}: {obj_url[:200]}...")

                        async with session.get(obj_url, ssl=ssl_context, timeout=30) as obj_response:
                            if obj_response.status == 200:
                                obj_xml = await obj_response.text()

                                try:
                                    obj_root = ET.fromstring(obj_xml)
                                    obj_entry = obj_root.find('.//entry')

                                    if obj_entry is not None:
                                        # Extract IP value from different possible formats
                                        ip_netmask = obj_entry.find('ip-netmask')
                                        ip_range = obj_entry.find('ip-range')
                                        fqdn = obj_entry.find('fqdn')

                                        obj_type = None
                                        obj_value = None

                                        if ip_netmask is not None and ip_netmask.text:
                                            obj_type = "ip-netmask"
                                            obj_value = ip_netmask.text.strip()
                                        elif ip_range is not None and ip_range.text:
                                            obj_type = "ip-range"
                                            obj_value = ip_range.text.strip()
                                        elif fqdn is not None and fqdn.text:
                                            obj_type = "fqdn"
                                            obj_value = fqdn.text.strip()

                                        result["members"].append({
                                            "name": member_name,
                                            "type": obj_type,
                                            "value": obj_value,
                                            "location": group_location,
                                            "device_group": group_device_group
                                        })
                                        logger.debug(f"\u2713 Found address object '{member_name}': {obj_type}={obj_value}")

                                except ET.ParseError as e:
                                    logger.debug(f"Error parsing address object '{member_name}' XML: {e}")
                                    # Still add the member name even if we can't get the value
                                    result["members"].append({
                                        "name": member_name,
                                        "type": "unknown",
                                        "value": None,
                                        "location": group_location,
                                        "device_group": group_device_group
                                    })
                            else:
                                logger.debug(f"Address object '{member_name}' query failed with status {obj_response.status}")
                                # Still add the member name even if we can't get the value
                                result["members"].append({
                                    "name": member_name,
                                    "type": "unknown",
                                    "value": None,
                                    "location": group_location,
                                    "device_group": group_device_group
                                })

                    except Exception as e:
                        logger.debug(f"Error querying address object '{member_name}': {str(e)}")
                        # Still add the member name even if we can't get the value
                        result["members"].append({
                            "name": member_name,
                            "type": "unknown",
                            "value": None,
                            "location": group_location,
                            "device_group": group_device_group
                        })
            else:
                logger.debug(f"Address group '{address_group_name}' has no static members")

            # Step 2: Query policies (security and NAT) that use this address group
            result["policies"] = []
            policies_by_group = {}  # Track policies per address group

            if found_group:
                logger.debug(f"Querying policies for address group '{address_group_name}'")

                # Query policies from the location where the group was found
                location_type = group_location
                location_name = group_device_group

                try:
                    # Query Security Policies - both Pre and Post Rules
                    security_rulebases = ["pre-rulebase", "post-rulebase"]

                    for rulebase in security_rulebases:
                        if location_type == "device-group":
                            sec_xpath = f"/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='{location_name}']/{rulebase}/security/rules/entry"
                        else:  # shared
                            sec_xpath = f"/config/shared/{rulebase}/security/rules/entry"

                        sec_url = f"{panorama_url}/api/?type=config&action=get&xpath={urllib.parse.quote(sec_xpath)}&key={api_key}"
                        logger.debug(f"Querying security policies from {location_type} {rulebase}: {sec_url[:200]}...")

                        async with session.get(sec_url, ssl=ssl_context, timeout=30) as sec_response:
                            if sec_response.status == 200:
                                sec_xml = await sec_response.text()
                                logger.debug(f"Security policies XML response length: {len(sec_xml)} chars")
                                try:
                                    sec_root = ET.fromstring(sec_xml)
                                    sec_entries = sec_root.findall('.//entry')
                                    logger.debug(f"Found {len(sec_entries)} security policy entries in {rulebase} for {location_type} {location_name or 'shared'}")

                                    for entry in sec_entries:
                                        rule_name = entry.get('name')
                                        if not rule_name:
                                            continue

                                        # Check source and destination for address group references
                                        source = entry.find('source')
                                        destination = entry.find('destination')

                                        # Get source and destination members for checking
                                        source_members_list = source.findall('member') if source is not None else []
                                        dest_members_list = destination.findall('member') if destination is not None else []
                                        source_members = [m.text for m in source_members_list if m.text]
                                        dest_members = [m.text for m in dest_members_list if m.text]

                                        # Check if the address group is referenced
                                        source_has_group = any(m == address_group_name for m in source_members)
                                        dest_has_group = any(m == address_group_name for m in dest_members)

                                        if source_has_group or dest_has_group:
                                            # Get action
                                            action_elem = entry.find('action')
                                            action = action_elem.text if action_elem is not None else "unknown"

                                            # Get service
                                            service_elem = entry.find('service')
                                            services = [s.text for s in service_elem.findall('member')] if service_elem is not None else []

                                            policy_key = f"{location_type}:{location_name or 'shared'}:{rule_name}:{rulebase}"
                                            if policy_key not in policies_by_group:
                                                policies_by_group[policy_key] = {
                                                    "name": rule_name,
                                                    "type": "security",
                                                    "rulebase": rulebase,
                                                    "location": location_type,
                                                    "device_group": location_name if location_type == "device-group" else None,
                                                    "action": action,
                                                    "source": source_members,
                                                    "destination": dest_members,
                                                    "services": services,
                                                    "address_groups": [address_group_name]
                                                }

                                            logger.debug(f"Found security policy '{rule_name}' ({rulebase}) using address group '{address_group_name}' in {location_type} {location_name or 'shared'}")

                                except ET.ParseError as e:
                                    logger.debug(f"Error parsing security policies XML from {rulebase}: {e}")
                            elif sec_response.status == 404:
                                logger.debug(f"No security policies found in {rulebase} for {location_type} {location_name or 'shared'} (404)")
                            else:
                                logger.debug(f"Security policies query failed with status {sec_response.status} for {rulebase}")

                    # Query NAT Policies - both Pre and Post Rules
                    nat_rulebases = ["pre-rulebase", "post-rulebase"]

                    for rulebase in nat_rulebases:
                        if location_type == "device-group":
                            nat_xpath = f"/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='{location_name}']/{rulebase}/nat/rules/entry"
                        else:  # shared
                            nat_xpath = f"/config/shared/{rulebase}/nat/rules/entry"

                        nat_url = f"{panorama_url}/api/?type=config&action=get&xpath={urllib.parse.quote(nat_xpath)}&key={api_key}"
                        logger.debug(f"Querying NAT policies from {location_type} {rulebase}: {nat_url[:200]}...")

                        async with session.get(nat_url, ssl=ssl_context, timeout=30) as nat_response:
                            if nat_response.status == 200:
                                nat_xml = await nat_response.text()
                                try:
                                    nat_root = ET.fromstring(nat_xml)
                                    nat_entries = nat_root.findall('.//entry')

                                    for entry in nat_entries:
                                        rule_name = entry.get('name')
                                        if not rule_name:
                                            continue

                                        # Check source-translation and destination-translation
                                        source_translation = entry.find('source-translation')
                                        destination_translation = entry.find('destination-translation')

                                        # Also check source and destination
                                        source = entry.find('source')
                                        destination = entry.find('destination')

                                        # Get source and destination members for checking
                                        source_members_list = source.findall('member') if source is not None else []
                                        dest_members_list = destination.findall('member') if destination is not None else []
                                        source_members = [m.text for m in source_members_list if m.text]
                                        dest_members = [m.text for m in dest_members_list if m.text]

                                        found_in_nat = False
                                        nat_type = None

                                        # Check source
                                        if any(m == address_group_name for m in source_members):
                                            found_in_nat = True
                                            nat_type = "source"

                                        # Check destination
                                        if any(m == address_group_name for m in dest_members):
                                            found_in_nat = True
                                            nat_type = "destination" if nat_type is None else f"{nat_type}/destination"

                                        # Check source-translation
                                        if source_translation is not None:
                                            static_ip = source_translation.find('static-ip')
                                            if static_ip is not None:
                                                translated_addr = static_ip.find('translated-address')
                                                if translated_addr is not None and translated_addr.text == address_group_name:
                                                    found_in_nat = True
                                                    nat_type = "source-translation" if nat_type is None else f"{nat_type}/source-translation"

                                        # Check destination-translation
                                        if destination_translation is not None:
                                            static_ip = destination_translation.find('static-ip')
                                            if static_ip is not None:
                                                translated_addr = static_ip.find('translated-address')
                                                if translated_addr is not None and translated_addr.text == address_group_name:
                                                    found_in_nat = True
                                                    nat_type = "destination-translation" if nat_type is None else f"{nat_type}/destination-translation"

                                        if found_in_nat:
                                            # Get service
                                            service_elem = entry.find('service')
                                            services = [s.text for s in service_elem.findall('member')] if service_elem is not None else []

                                            policy_key = f"{location_type}:{location_name or 'shared'}:{rule_name}:{rulebase}"
                                            if policy_key not in policies_by_group:
                                                policies_by_group[policy_key] = {
                                                    "name": rule_name,
                                                    "type": "nat",
                                                    "rulebase": rulebase,
                                                    "location": location_type,
                                                    "device_group": location_name if location_type == "device-group" else None,
                                                    "nat_type": nat_type,
                                                    "source": source_members,
                                                    "destination": dest_members,
                                                    "services": services,
                                                    "address_groups": [address_group_name]
                                                }

                                            logger.debug(f"Found NAT policy '{rule_name}' ({rulebase}) using address group '{address_group_name}' in {location_type} {location_name or 'shared'}")

                                except ET.ParseError as e:
                                    logger.debug(f"Error parsing NAT policies XML from {rulebase}: {e}")
                            elif nat_response.status == 404:
                                logger.debug(f"No NAT policies found in {rulebase} for {location_type} {location_name or 'shared'} (404)")
                            else:
                                logger.debug(f"NAT policies query failed with status {nat_response.status} for {rulebase}")

                except Exception as e:
                    logger.debug(f"Error querying policies from {location_type}: {str(e)}")

                # Convert policies dict to list
                result["policies"] = list(policies_by_group.values())
                logger.debug(f"Found {len(result['policies'])} policies using address group '{address_group_name}'")

        # Send result to LLM for analysis and table format summary
        llm = _get_llm()
        if llm is not None and "error" not in result:
            logger.debug(f"LLM available, starting analysis for Panorama address group members query")
            try:
                from langchain_core.prompts import ChatPromptTemplate

                system_prompt = """You are a network security assistant. Analyze the Panorama address group members query results and provide a concise summary in TABLE FORMAT.

Provide a summary that includes:
- The queried address group name
- All address objects in the group (name, type, IP value/CIDR)
- Location where the group was found (shared or device group)
- Total count of members
- **For the address group, list ALL policies (security and NAT) that use that address group**
  - Include policy name, type (security/NAT), action (for security policies), source, destination, services

Format your response as markdown tables:

**Table 1: Object group details**
Columns: Address Object Name, Type, IP Address/Value, Location

**Table 2: Policy details**
Columns: Address Group, Policy Name, Policy Type, Rulebase (Pre/Post), Action/NAT Type, Source, Destination, Services, Location

**IMPORTANT: When displaying policies, clearly show which address group each policy uses, along with the policy details (action, source, destination, services).**

Keep the summary concise and informative. Focus on the key findings."""

                analysis_prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "Analyze this Panorama address group members query result:\n{query_result}")
                ])

                formatted_messages = analysis_prompt_template.format_messages(
                    query_result=json.dumps(result, indent=2)
                )

                logger.debug(f"Invoking LLM for Panorama analysis")
                response = await asyncio.wait_for(
                    llm.ainvoke(formatted_messages),
                    timeout=30.0
                )
                content = response.content if hasattr(response, 'content') else str(response)

                logger.debug(f"LLM response received: {content[:200]}...")

                result["ai_analysis"] = {"summary": content}

            except Exception as e:
                logger.debug(f"LLM analysis failed: {str(e)}")
                import traceback
                logger.debug(f"LLM analysis traceback: {traceback.format_exc()}")
                # Provide a basic summary if LLM fails
                members_count = len(result.get("members", []))
                if members_count > 0:
                    result["ai_analysis"] = {
                        "summary": f"Address group '{address_group_name}' contains {members_count} member(s)."
                    }
                else:
                    result["ai_analysis"] = {
                        "summary": f"Address group '{address_group_name}' has no members or was not found."
                    }

    except Exception as e:
        logger.error(f"Exception querying Panorama for address group members: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        result["error"] = f"Error querying Panorama: {str(e)}"

    return result
