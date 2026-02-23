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
import time as _time
import aiohttp
import xml.etree.ElementTree as ET
import urllib.parse
import ipaddress
from typing import Optional, Dict, Any, List

from tools.shared import mcp, _get_llm, ChatPromptTemplate, setup_logging
import panoramaauth

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Module-level caches (shared across requests, TTL = 5 minutes)
# ---------------------------------------------------------------------------

_CACHE_TTL = 300.0

# Device group list cache: (list[str], timestamp) | None
_dg_cache: tuple[list, float] | None = None

# Address object cache: "loc_type:loc_name" -> ({name: {type, value}}, timestamp)
_addr_obj_cache: dict[str, tuple[dict, float]] = {}


def _parse_address_entries(entries) -> dict:
    """Parse XML address <entry> elements into {name: {type, value}} dict."""
    result = {}
    for entry in entries:
        name = entry.get('name')
        if not name:
            continue
        ip_netmask = entry.find('ip-netmask')
        ip_range = entry.find('ip-range')
        fqdn_el = entry.find('fqdn')
        if ip_netmask is not None and ip_netmask.text:
            result[name] = {"type": "ip-netmask", "value": ip_netmask.text.strip()}
        elif ip_range is not None and ip_range.text:
            result[name] = {"type": "ip-range", "value": ip_range.text.strip()}
        elif fqdn_el is not None and fqdn_el.text:
            result[name] = {"type": "fqdn", "value": fqdn_el.text.strip()}
        else:
            result[name] = {"type": None, "value": None}
    return result


async def _get_device_groups_cached(session, panorama_url: str, api_key: str, ssl_context) -> list:
    """Return device group names from Panorama, refreshed every _CACHE_TTL seconds."""
    global _dg_cache
    now = _time.monotonic()
    if _dg_cache and (now - _dg_cache[1]) < _CACHE_TTL:
        logger.debug("Device groups: cache hit (%d groups)", len(_dg_cache[0]))
        return _dg_cache[0]

    names: list = []
    try:
        url = (
            f"{panorama_url}/api/?type=config&action=get"
            f"&xpath=/config/devices/entry[@name='localhost.localdomain']/device-group/entry"
            f"&key={api_key}"
        )
        async with session.get(url, ssl=ssl_context, timeout=15) as resp:
            if resp.status == 200:
                root = ET.fromstring(await resp.text())
                # Direct children only — .//entry would also match address/policy
                # entries nested inside device groups, producing bogus DG names.
                names = [e.get('name') for e in root.findall('./result/entry') if e.get('name')]
                logger.debug("Device groups: fetched %d names from Panorama", len(names))
    except Exception as exc:
        logger.debug("Failed to fetch device group list: %s", exc)

    _dg_cache = (names, now)
    return names


async def _get_address_objects_cached(
    session, panorama_url: str, api_key: str, ssl_context,
    location_type: str, location_name: Optional[str],
) -> dict:
    """Return all address objects for a location, refreshed every _CACHE_TTL seconds."""
    key = f"{location_type}:{location_name or 'shared'}"
    global _addr_obj_cache
    now = _time.monotonic()
    cached = _addr_obj_cache.get(key)
    if cached and (now - cached[1]) < _CACHE_TTL:
        logger.debug("Address objects: cache hit for %s (%d objects)", key, len(cached[0]))
        return cached[0]

    if location_type == "device-group":
        xpath = (
            f"/config/devices/entry[@name='localhost.localdomain']"
            f"/device-group/entry[@name='{location_name}']/address"
        )
    else:
        xpath = "/config/shared/address"

    url = f"{panorama_url}/api/?type=config&action=get&xpath={urllib.parse.quote(xpath)}&key={api_key}"
    objects: dict = {}
    try:
        async with session.get(url, ssl=ssl_context, timeout=30) as resp:
            if resp.status == 200:
                root = ET.fromstring(await resp.text())
                objects = _parse_address_entries(root.findall('.//entry'))
                logger.debug("Address objects: fetched %d from %s", len(objects), key)
    except Exception as exc:
        logger.debug("Failed to fetch address objects for %s: %s", key, exc)

    _addr_obj_cache[key] = (objects, now)
    return objects


async def _fetch_address_groups_for_location(
    session, panorama_url: str, api_key: str, ssl_context,
    location_type: str, location_name: Optional[str],
) -> list:
    """Fetch all address group entries for one Panorama location. Returns list of dicts."""
    if location_type == "device-group":
        xpath = (
            f"/config/devices/entry[@name='localhost.localdomain']"
            f"/device-group/entry[@name='{location_name}']/address-group"
        )
    else:
        xpath = "/config/shared/address-group"

    url = f"{panorama_url}/api/?type=config&action=get&xpath={urllib.parse.quote(xpath)}&key={api_key}"
    groups = []
    try:
        async with session.get(url, ssl=ssl_context, timeout=30) as resp:
            if resp.status == 200:
                root = ET.fromstring(await resp.text())
                for entry in root.findall('.//entry'):
                    gname = entry.get('name')
                    if not gname:
                        continue
                    static = entry.find('static')
                    members = [m.text for m in static.findall('member') if m.text] if static is not None else []
                    groups.append({
                        "name": gname,
                        "members": members,
                        "location_type": location_type,
                        "location_name": location_name,
                    })
    except Exception as exc:
        logger.debug("Failed to fetch address groups for %s %s: %s", location_type, location_name, exc)
    return groups


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
    Find which Panorama address groups contain a given IP address.

    Use for: queries with an IP address (has dots, e.g. "10.0.0.1") asking which address group/object group it belongs to.
    Input is an IP; output is the list of groups that contain it.
    Do NOT use for: device names (have dashes, use get_device_rack_location), group member listing (use query_panorama_address_group_members), path/traffic checks.

    Examples:
    - "what address group is 10.0.0.1 in?" → ip_address="10.0.0.1"
    - "which group contains 11.0.0.0/24?" → ip_address="11.0.0.0/24"
    - "find address group for 192.168.1.5" → ip_address="192.168.1.5"

    If query is just a bare IP with no context, ask: "What would you like to do with [IP]?
    1) Query Panorama for object groups  2) Query network path"

    **HANDLING FOLLOW-UP RESPONSES:**
    - If conversation history shows a previous clarification question was asked in the standard format: "What would you like to do with [IP]? 1) Query Panorama for object groups, 2) Query network path"
    - AND the current query is just "1" → user selected option 1 (Panorama) → use this tool with ip_address from earlier history.

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
            "error": "Failed to authenticate with Panorama. Ensure PANORAMA_USERNAME and PANORAMA_PASSWORD are set (or in Azure Key Vault as PANORAMA-USERNAME / PANORAMA-PASSWORD) and the MCP server was started with access to them."
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
            # Step 1: Build locations list, then fetch all address objects in parallel
            locations = []
            if device_group:
                locations.append(("device-group", device_group))
            else:
                locations.append(("shared", None))
                dg_names = await _get_device_groups_cached(session, panorama_url, api_key, ssl_context)
                for dg_name in dg_names:
                    locations.append(("device-group", dg_name))
                logger.debug("Searching %d location(s): shared + %d device group(s)", len(locations), len(dg_names))

            # Parallel fetch of address objects for all locations
            addr_obj_results = await asyncio.gather(
                *[_get_address_objects_cached(session, panorama_url, api_key, ssl_context, lt, ln)
                  for lt, ln in locations],
                return_exceptions=True,
            )

            # Build location->(objects dict) map and filter by queried IP
            location_objects: Dict[tuple, Dict] = {}
            matching_address_objects = []
            for (lt, ln), obj_result in zip(locations, addr_obj_results):
                if isinstance(obj_result, Exception):
                    logger.debug("Error fetching address objects for %s %s: %s", lt, ln, obj_result)
                    continue
                location_objects[(lt, ln)] = obj_result

                for obj_name, details in obj_result.items():
                    obj_type = details.get("type")
                    obj_value = details.get("value")
                    if not obj_type or not obj_value:
                        continue

                    matches = False
                    logger.debug("Checking '%s' (%s: %s) in %s %s", obj_name, obj_type, obj_value, lt, ln or "shared")
                    try:
                        if obj_type == "ip-netmask":
                            if '/' in obj_value:
                                obj_network = ipaddress.ip_network(obj_value, strict=False)
                                if is_cidr:
                                    matches = (query_network == obj_network) or query_network.overlaps(obj_network)
                                    logger.debug("CIDR vs CIDR: query=%s obj=%s match=%s", query_network, obj_network, matches)
                                else:
                                    matches = query_ip in obj_network
                                    logger.debug("IP in CIDR: %s in %s = %s", query_ip, obj_network, matches)
                            else:
                                obj_ip = ipaddress.ip_address(obj_value)
                                if is_cidr:
                                    matches = obj_ip in query_network
                                    logger.debug("CIDR contains IP: %s in %s = %s", obj_ip, query_network, matches)
                                else:
                                    matches = (query_ip == obj_ip)
                                    logger.debug("IP vs IP: %s == %s = %s", query_ip, obj_ip, matches)
                        elif obj_type == "ip-range":
                            if '-' in obj_value:
                                start_str, end_str = obj_value.split('-', 1)
                                start = ipaddress.ip_address(start_str.strip())
                                end = ipaddress.ip_address(end_str.strip())
                                matches = start <= query_ip <= end
                                logger.debug("IP in range %s-%s = %s", start, end, matches)
                        # fqdn never matches an IP directly
                    except (ValueError, ipaddress.AddressValueError) as exc:
                        logger.debug("Comparison error for %s: %s", obj_name, exc)

                    if matches:
                        matching_address_objects.append({
                            "name": obj_name,
                            "type": obj_type,
                            "value": obj_value,
                            "location": lt,
                            "device_group": ln if lt == "device-group" else None,
                        })
                        logger.debug("✓ MATCH: %s (%s: %s) in %s %s", obj_name, obj_type, obj_value, lt, ln or "shared")

            result["address_objects"] = matching_address_objects
            logger.debug("Found %d matching address object(s) across %d location(s)", len(matching_address_objects), len(locations))

            # Step 2: Fetch all address groups in parallel, resolve member details from cache (no N+1)
            # Only query locations that have matching address objects (reduces traffic)
            relevant_locations = (
                list({(ao["location"], ao["device_group"]) for ao in matching_address_objects})
                if matching_address_objects else locations
            )
            addr_grp_results = await asyncio.gather(
                *[_fetch_address_groups_for_location(session, panorama_url, api_key, ssl_context, lt, ln)
                  for lt, ln in relevant_locations],
                return_exceptions=True,
            )

            matching_obj_names = {ao["name"] for ao in matching_address_objects}
            for (lt, ln), grp_result in zip(relevant_locations, addr_grp_results):
                if isinstance(grp_result, Exception):
                    logger.debug("Error fetching address groups for %s %s: %s", lt, ln, grp_result)
                    continue
                loc_objs = location_objects.get((lt, ln), {})
                logger.debug("Found %d address groups in %s %s", len(grp_result), lt, ln or "shared")

                for grp in grp_result:
                    gname = grp["name"]
                    member_names = grp["members"]

                    # Check if any matching address object is a member of this group
                    matching_member = next((m for m in member_names if m in matching_obj_names), None)
                    if matching_member is None:
                        continue

                    # Resolve member details from cached address objects — no per-member HTTP call
                    group_members = []
                    for member_name in member_names:
                        det = loc_objs.get(member_name, {})
                        group_members.append({
                            "name": member_name,
                            "type": det.get("type"),
                            "value": det.get("value"),
                            "location": lt,
                            "device_group": ln if lt == "device-group" else None,
                        })
                        logger.debug("✓ Member '%s': %s=%s", member_name, det.get("type"), det.get("value"))

                    result["address_groups"].append({
                        "name": gname,
                        "location": lt,
                        "device_group": ln if lt == "device-group" else None,
                        "contains_address_object": matching_member,
                        "members": group_members,
                    })
                    logger.debug("Found address group '%s' containing '%s' with %d members", gname, matching_member, len(group_members))

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

                                            # Check source-translation (for groups only)
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
                                                        "rulebase": rulebase,
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
    List all IP addresses and address objects that are members of a named Panorama address group.

    Use for: queries with an ADDRESS GROUP NAME asking what IPs/members are in it.
    Input is a group name; output is the list of member IPs/objects.
    Do NOT use for: finding which group an IP belongs to (use query_panorama_ip_object_group), path/traffic checks, device/rack lookups.

    Examples:
    - "what IPs are in address group leander_web?" → address_group_name="leander_web"
    - "list members of group web_servers" → address_group_name="web_servers"
    - "what addresses are in dmz_hosts?" → address_group_name="dmz_hosts"

    Args:
        address_group_name: Name of the address group to query (e.g., "leander_web", "web_servers")
        device_group: Optional device group name to search within (if None, searches shared and all device groups)
        vsys: VSYS name (default: "vsys1")
    """
    import xml.etree.ElementTree as ET
    import urllib.parse

    logger.debug(f"query_panorama_address_group_members called with address_group_name={address_group_name}, device_group={device_group}, vsys={vsys}")

    # Get API key from panoramaauth
    api_key = await panoramaauth.get_api_key()
    if not api_key:
        return {
            "address_group_name": address_group_name,
            "error": "Failed to authenticate with Panorama. Ensure PANORAMA_USERNAME and PANORAMA_PASSWORD are set (or in Azure Key Vault as PANORAMA-USERNAME / PANORAMA-PASSWORD) and the MCP server was started with access to them."
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
                dg_names = await _get_device_groups_cached(session, panorama_url, api_key, ssl_context)
                for dg_name in dg_names:
                    locations.append(("device-group", dg_name))
                logger.debug("Searching %d location(s): shared + %d device group(s)", len(locations), len(dg_names))

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

                # Resolve member details from cache — no per-member HTTP call (eliminates N+1)
                loc_objs = await _get_address_objects_cached(
                    session, panorama_url, api_key, ssl_context, group_location, group_device_group
                )
                for member_name in member_names:
                    det = loc_objs.get(member_name, {})
                    result["members"].append({
                        "name": member_name,
                        "type": det.get("type"),
                        "value": det.get("value"),
                        "location": group_location,
                        "device_group": group_device_group,
                    })
                    logger.debug("✓ Member '%s': %s=%s", member_name, det.get("type"), det.get("value"))
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
