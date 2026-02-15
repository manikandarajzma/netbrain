"""
Panorama API Authentication Module
Handles authentication with Palo Alto Panorama API using API key.

This module provides:
- API key-based authentication for Panorama REST API
- Functions to query security zones for firewall interfaces
- SSL certificate verification bypass for self-signed certificates
"""

# Import aiohttp for asynchronous HTTP client operations
import aiohttp

# Import json module for JSON serialization
import json

# Import logging for structured log output
import logging

# Import ssl for SSL/TLS context configuration
import ssl

# Import os module for accessing environment variables
import os

# Import urllib.parse for URL encoding
import urllib.parse

# Import xml.etree.ElementTree for parsing XML responses
import xml.etree.ElementTree as ET

# Import Optional type hint from typing module
from typing import Optional, Dict, Any, List

# Load .env file if available
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

# Configure module logger
logger = logging.getLogger("netbrain.panoramaauth")

# Panorama API configuration
PANORAMA_URL = os.getenv("PANORAMA_URL", "https://192.168.15.247")

# Panorama username for authentication
PANORAMA_USERNAME = os.getenv("PANORAMA_USERNAME", "")

# Panorama password for authentication
PANORAMA_PASSWORD = os.getenv("PANORAMA_PASSWORD", "")

# Cache for API key
# Module-level variable to store the cached API key
# Underscore prefix indicates it's a private module-level variable
_api_key: Optional[str] = None


async def get_api_key() -> Optional[str]:
    """
    Get Panorama API key using username/password authentication.
    Returns cached key if available, otherwise requests a new one.

    This function implements key caching to minimize API calls:
    - Checks if a cached key exists
    - If available, returns the cached key immediately
    - If missing, requests a new key from the keygen endpoint
    - Caches the new key for reuse

    Returns:
        Optional[str]: API key string if successful, None if authentication fails
    """
    global _api_key

    # Check if we have a cached key
    if _api_key:
        return _api_key

    # Validate that USERNAME and PASSWORD are configured
    if not PANORAMA_USERNAME or not PANORAMA_PASSWORD:
        logger.warning("Panorama USERNAME or PASSWORD not configured")
        return None

    # URL encode password to handle special characters
    password_encoded = urllib.parse.quote(PANORAMA_PASSWORD, safe='')

    # Construct the Panorama keygen API endpoint URL
    keygen_url = f"{PANORAMA_URL}/api/?type=keygen&user={PANORAMA_USERNAME}&password={password_encoded}"

    logger.debug(f"Attempting to get API key from: {PANORAMA_URL}")
    logger.debug(f"Username: {PANORAMA_USERNAME}")
    logger.debug(f"Keygen URL (password hidden): {PANORAMA_URL}/api/?type=keygen&user={PANORAMA_USERNAME}&password=***")

    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(keygen_url, ssl=ssl_context, timeout=10) as response:
                if response.status == 200:
                    # Parse XML response
                    response_text = await response.text()
                    try:
                        root = ET.fromstring(response_text)
                        status = root.attrib.get('status')

                        if status == 'success':
                            key_element = root.find('.//key')
                            if key_element is not None and key_element.text:
                                _api_key = key_element.text
                                logger.debug("Panorama API key retrieved successfully")
                                return _api_key
                            else:
                                logger.warning("API key not found in Panorama response")
                                return None
                        else:
                            # Extract error message if available
                            msg_element = root.find('.//msg')
                            error_msg = msg_element.text if msg_element is not None else "Unknown error"
                            logger.error(f"Panorama authentication failed: {error_msg}")
                            return None
                    except ET.ParseError as e:
                        logger.error(f"Error parsing Panorama XML response: {e}")
                        return None
                else:
                    logger.error(f"Get Panorama API key failed! HTTP {response.status}: {await response.text()}")
                    return None
    except Exception as e:
        logger.error(f"Error getting Panorama API key: {e}")
        return None


def clear_api_key_cache():
    """
    Clear the cached API key (useful for testing or forced re-authentication).

    This function resets the key cache, forcing the next call to get_api_key()
    to request a fresh key from the API.
    """
    global _api_key
    _api_key = None


async def get_security_zones_for_interface(
    firewall_serial: Optional[str] = None,
    firewall_name: Optional[str] = None,
    interface_name: str = "",
    template: Optional[str] = None,
    vsys: str = "vsys1"
) -> Optional[Dict[str, Any]]:
    """
    Get security zone information for a specific firewall interface.

    Args:
        firewall_serial: Serial number of the firewall (for Panorama-managed devices)
        firewall_name: Name of the firewall device (alternative to serial)
        interface_name: Name of the interface to query (e.g., "ethernet1/1", "Ethernet1")
        template: Template name if zones are template-based (optional)
        vsys: VSYS name (default: "vsys1")

    Returns:
        dict: Zone information including zone name, or None if not found/error
    """
    # Get API key automatically (will use cached key if available)
    api_key = await get_api_key()
    if not api_key:
        logger.warning("Could not retrieve Panorama API key")
        return None

    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    headers = {
        "X-PAN-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Try multiple approaches to find the zone
    # Approach 1: Query zones from template (if template provided)
    if template:
        try:
            url = f"{PANORAMA_URL}/restapi/v11.0/network/zones?location=template&template={template}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, ssl=ssl_context) as response:
                    if response.status == 200:
                        zones_data = await response.json()
                        # Parse zones to find which zone contains this interface
                        if isinstance(zones_data, dict) and "result" in zones_data:
                            for zone_entry in zones_data.get("result", {}).get("entry", []):
                                if isinstance(zone_entry, dict):
                                    members = zone_entry.get("network", {}).get("layer3", {}).get("member", [])
                                    if isinstance(members, list):
                                        # Check if interface is in members list
                                        for member in members:
                                            if interface_name.lower() in str(member).lower():
                                                return {
                                                    "zone_name": zone_entry.get("@name", "Unknown"),
                                                    "interface": interface_name,
                                                    "source": "template"
                                                }
        except Exception as e:
            logger.debug(f"Error querying template zones: {str(e)}")

    # Approach 2: Query zones from firewall directly (if firewall serial/name provided)
    if firewall_serial or firewall_name:
        try:
            # First, get list of devices from Panorama
            # Use API key as query parameter for XML API
            devices_url = f"{PANORAMA_URL}/api/?type=op&cmd=<show><devices><all></all></devices></show>&key={api_key}"
            async with aiohttp.ClientSession() as session:
                async with session.get(devices_url, ssl=ssl_context) as response:
                    if response.status == 200:
                        devices_data = await response.text()
                        # Parse XML response to find firewall serial/name
                        # Then query zones for that firewall

                        # Query zones via Panorama targeting the firewall
                        target_param = f"&target={firewall_serial}" if firewall_serial else ""
                        zones_url = f"{PANORAMA_URL}/api/?type=config&action=get&xpath=/config/devices/entry[@name='localhost.localdomain']/network/zone&key={api_key}{target_param}"

                        async with session.get(zones_url, ssl=ssl_context) as response:
                            if response.status == 200:
                                zones_xml = await response.text()
                                # Parse XML to find zone with matching interface
                                # This is a simplified version - full XML parsing would be needed
                                pass
        except Exception as e:
            logger.debug(f"Error querying firewall zones: {str(e)}")

    # Approach 3: Query zones from VSYS (local firewall configuration)
    try:
        url = f"{PANORAMA_URL}/restapi/v11.0/network/zones?location=vsys&vsys={vsys}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, ssl=ssl_context) as response:
                if response.status == 200:
                    zones_data = await response.json()
                    # Parse zones to find which zone contains this interface
                    if isinstance(zones_data, dict) and "result" in zones_data:
                        for zone_entry in zones_data.get("result", {}).get("entry", []):
                            if isinstance(zone_entry, dict):
                                members = zone_entry.get("network", {}).get("layer3", {}).get("member", [])
                                if isinstance(members, list):
                                    # Check if interface is in members list
                                    for member in members:
                                        if interface_name.lower() in str(member).lower():
                                            return {
                                                "zone_name": zone_entry.get("@name", "Unknown"),
                                                "interface": interface_name,
                                                "source": "vsys"
                                            }
    except Exception as e:
        logger.debug(f"Error querying VSYS zones: {str(e)}")

    return None


async def get_zones_for_firewall_interfaces(
    firewall_name: str,
    interfaces: List[str],
    firewall_serial: Optional[str] = None,
    template: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    Get security zones for multiple firewall interfaces.

    Args:
        firewall_name: Name of the firewall device
        interfaces: List of interface names (e.g., ["ethernet1/1", "ethernet1/2"])
        firewall_serial: Serial number of the firewall (optional, for Panorama-managed devices)
        template: Template name if zones are template-based (optional)

    Returns:
        dict: Mapping of interface name to zone name (e.g., {"ethernet1/1": "trust", "ethernet1/2": "untrust"})
    """
    result = {}

    # Initialize result with None for all interfaces
    for interface in interfaces:
        result[interface] = None

    # Get API key automatically (will use cached key if available)
    api_key = await get_api_key()
    if not api_key:
        logger.error(f"Could not retrieve API key for {firewall_name}")
        return result

    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    headers = {
        "X-PAN-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Use "Global" template if not specified
    template_name = template or "Global"
    logger.debug(f"Querying zones from template '{template_name}' for firewall '{firewall_name}'")
    logger.debug(f"Interfaces to query: {interfaces}")

    # Try different REST API versions (most common: v9.0, v9.1, v10.0, v10.1, v10.2)
    api_versions = ["v10.2", "v10.1", "v10.0", "v9.1", "v9.0"]

    # Query zones from template
    try:
        async with aiohttp.ClientSession() as session:
            zones_found = False
            for api_version in api_versions:
                # Try both with and without vsys parameter
                url_variants = [
                    f"{PANORAMA_URL}/restapi/{api_version}/network/zones?location=template&template={template_name}",
                    f"{PANORAMA_URL}/restapi/{api_version}/network/zones?location=template&template={template_name}&vsys=vsys1"
                ]

                for url in url_variants:
                    logger.debug(f"Trying API version {api_version}: {url}")

                    async with session.get(url, headers=headers, ssl=ssl_context, timeout=30) as response:
                        logger.debug(f"API Response Status: {response.status}")

                        if response.status == 501:
                            # Version not supported, try next version
                            response_text = await response.text()
                            logger.debug(f"Version {api_version} not supported: {response_text[:200]}")
                            continue

                        if response.status != 200:
                            # Other error, try next version
                            response_text = await response.text()
                            logger.debug(f"API call failed with version {api_version}: HTTP {response.status}: {response_text[:200]}")
                            continue

                        # Success! Parse the response
                        zones_data = await response.json()
                        logger.debug(f"Zones data received (version {api_version}): {json.dumps(zones_data, indent=2)[:500]}...")

                        # Parse zones to find which zone contains each interface
                        if isinstance(zones_data, dict) and "result" in zones_data:
                            zones_result = zones_data.get("result", {})

                            # Handle different response structures
                            zone_entries = []
                            if isinstance(zones_result, dict):
                                if "entry" in zones_result:
                                    zone_entries = zones_result["entry"] if isinstance(zones_result["entry"], list) else [zones_result["entry"]]
                                elif isinstance(zones_result, list):
                                    zone_entries = zones_result

                            logger.debug(f"Found {len(zone_entries)} zones in template")

                            # Build a mapping of interface -> zone
                            interface_to_zone = {}

                            for zone_entry in zone_entries:
                                if not isinstance(zone_entry, dict):
                                    continue

                                zone_name = zone_entry.get("@name") or zone_entry.get("name")
                                if not zone_name:
                                    continue

                                # Extract interface members from zone
                                network = zone_entry.get("network", {})
                                layer3 = network.get("layer3", {}) if isinstance(network, dict) else {}
                                members = layer3.get("member", []) if isinstance(layer3, dict) else []

                                # Handle both list and single value
                                if not isinstance(members, list):
                                    members = [members] if members else []

                                logger.debug(f"Zone '{zone_name}' has {len(members)} interface members: {members}")

                                # Map each member interface to this zone
                                for member in members:
                                    if member:
                                        member_str = str(member).strip()
                                        # Store both exact and case-insensitive matches
                                        interface_to_zone[member_str] = zone_name
                                        interface_to_zone[member_str.lower()] = zone_name

                            # Match requested interfaces to zones
                            for interface in interfaces:
                                if not interface:
                                    continue

                                interface_str = str(interface).strip()
                                interface_lower = interface_str.lower()

                                # Try exact match first
                                if interface_str in interface_to_zone:
                                    result[interface] = interface_to_zone[interface_str]
                                    logger.debug(f"Matched '{interface}' to zone '{interface_to_zone[interface_str]}' (exact)")
                                elif interface_lower in interface_to_zone:
                                    result[interface] = interface_to_zone[interface_lower]
                                    logger.debug(f"Matched '{interface}' to zone '{interface_to_zone[interface_lower]}' (case-insensitive)")
                                else:
                                    # Try partial match (e.g., "ethernet1/1" matches "Ethernet1/1")
                                    matched = False
                                    for member_intf, zone in interface_to_zone.items():
                                        if interface_lower == member_intf.lower():
                                            result[interface] = zone
                                            logger.debug(f"Matched '{interface}' to zone '{zone}' (partial)")
                                            matched = True
                                            break

                                    if not matched:
                                        logger.debug(f"No zone found for interface '{interface}'")

                            zones_found = True
                            logger.debug(f"Successfully retrieved zones using API version {api_version}")
                            break  # Exit the url_variants loop on success

                        else:
                            logger.warning(f"Unexpected response structure: {type(zones_data)}")

                    if zones_found:
                        break  # Exit the api_versions loop on success

                if zones_found:
                    break  # Exit the api_versions loop on success

            if not zones_found:
                logger.error("Failed to retrieve zones with any supported API version")

    except Exception as e:
        logger.error(f"Exception querying zones: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

    logger.debug(f"Final zones for {firewall_name}: {result}")
    return result


async def get_device_groups_for_firewalls(
    firewall_names: List[str]
) -> Dict[str, Optional[str]]:
    """
    Get device groups for multiple firewalls.

    Args:
        firewall_names: List of firewall device names (e.g., ["roundrock-dc-fw1", "leander-dc-fw1"])

    Returns:
        dict: Mapping of firewall name to device group name (e.g., {"roundrock-dc-fw1": "DeviceGroup1"})
    """
    result = {}

    # Initialize result with None for all firewalls
    for fw_name in firewall_names:
        result[fw_name] = None

    # Get API key automatically (will use cached key if available)
    api_key = await get_api_key()
    if not api_key:
        logger.error("Could not retrieve API key for device group query")
        return result

    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    headers = {
        "X-PAN-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    logger.debug(f"Querying device groups for firewalls: {firewall_names}")

    try:
        async with aiohttp.ClientSession() as session:
            # First, query all devices to see their structure and find device groups
            devices_url = f"{PANORAMA_URL}/api/?type=op&cmd=<show><devices><all></all></devices></show>&key={api_key}"
            logger.debug(f"Querying all devices: {devices_url[:100]}...")

            async with session.get(devices_url, ssl=ssl_context, timeout=30) as devices_response:
                if devices_response.status == 200:
                    devices_xml = await devices_response.text()
                    logger.debug(f"Devices XML response length: {len(devices_xml)}")
                    logger.debug(f"Devices XML (first 2000 chars): {devices_xml[:2000]}...")

                    try:
                        root = ET.fromstring(devices_xml)

                        # Find all devices
                        devices = root.findall('.//entry')
                        logger.debug(f"Found {len(devices)} devices")

                        for device in devices:
                            # Get hostname from device (this is the actual firewall name)
                            hostname_elem = device.find('hostname')
                            device_hostname = hostname_elem.text if hostname_elem is not None else None

                            # Also get serial for reference
                            serial_elem = device.find('serial')
                            device_serial = serial_elem.text if serial_elem is not None else device.get('name')

                            if device_hostname:
                                logger.debug(f"Checking device: hostname='{device_hostname}', serial='{device_serial}'")

                                # Match firewall names using hostname (case-insensitive)
                                device_hostname_lower = device_hostname.lower()
                                matched_fw = None
                                for fw_name in firewall_names:
                                    if (fw_name.lower() == device_hostname_lower or
                                        device_hostname_lower in fw_name.lower() or
                                        fw_name.lower() in device_hostname_lower):
                                        matched_fw = fw_name
                                        logger.debug(f"Matched firewall name '{fw_name}' to device hostname '{device_hostname}'")
                                        break

                                if matched_fw:
                                    # Try to find device-group for this device
                                    # Use serial number for device config query
                                    device_config_url = f"{PANORAMA_URL}/api/?type=config&action=get&xpath=/config/devices/entry[@name='{device_serial}']&key={api_key}"
                                    logger.debug(f"Querying device config for '{device_hostname}' (serial: '{device_serial}')")

                                    async with session.get(device_config_url, ssl=ssl_context, timeout=30) as device_config_response:
                                        if device_config_response.status == 200:
                                            device_config_xml = await device_config_response.text()
                                            logger.debug(f"Device config XML (first 1000 chars): {device_config_xml[:1000]}...")

                                            # Try to find device-group in the config
                                            try:
                                                config_root = ET.fromstring(device_config_xml)
                                                # Look for device-group reference
                                                device_group_elem = config_root.find('.//device-group')
                                                if device_group_elem is not None:
                                                    group_name = device_group_elem.text or device_group_elem.get('name')
                                                    if group_name:
                                                        result[matched_fw] = group_name
                                                        logger.debug(f"Found device group '{group_name}' for firewall '{matched_fw}'")
                                            except ET.ParseError as e:
                                                logger.debug(f"Error parsing device config XML: {e}")

                                    # Query device-groups and check each one
                                    # First try operational command to list device groups
                                    show_dg_url = f"{PANORAMA_URL}/api/?type=op&cmd=<show><devicegroups></devicegroups></show>&key={api_key}"
                                    logger.debug("Querying device groups via operational command")

                                    async with session.get(show_dg_url, ssl=ssl_context, timeout=30) as show_dg_response:
                                        if show_dg_response.status == 200:
                                            show_dg_xml = await show_dg_response.text()
                                            logger.debug(f"Show device groups XML (first 2000 chars): {show_dg_xml[:2000]}...")

                                            try:
                                                show_dg_root = ET.fromstring(show_dg_xml)
                                                # Find all device groups in the operational response
                                                dg_entries = show_dg_root.findall('.//entry')
                                                logger.debug(f"Found {len(dg_entries)} device groups via operational command")

                                                # For each device group, query its devices
                                                for dg_entry in dg_entries:
                                                    dg_name = dg_entry.get('name') or dg_entry.text
                                                    if not dg_name:
                                                        continue

                                                    logger.debug(f"Checking device group '{dg_name}' for device '{device_serial}'")

                                                    # Query devices in this device group
                                                    dg_devices_url = f"{PANORAMA_URL}/api/?type=config&action=get&xpath=/config/device-group/entry[@name='{dg_name}']/devices&key={api_key}"
                                                    async with session.get(dg_devices_url, ssl=ssl_context, timeout=30) as dg_devices_response:
                                                        if dg_devices_response.status == 200:
                                                            dg_devices_xml = await dg_devices_response.text()
                                                            logger.debug(f"Devices in group '{dg_name}' XML: {dg_devices_xml[:1000]}...")

                                                            try:
                                                                dg_devices_root = ET.fromstring(dg_devices_xml)
                                                                devices_in_dg = dg_devices_root.findall('.//entry')
                                                                for dev in devices_in_dg:
                                                                    dev_serial = dev.get('name') or dev.text
                                                                    if dev_serial and (dev_serial == device_serial or dev_serial.lower() == device_serial.lower()):
                                                                        result[matched_fw] = dg_name
                                                                        logger.debug(f"Matched firewall '{matched_fw}' to device group '{dg_name}' via operational command")
                                                                        break
                                                            except ET.ParseError as e:
                                                                logger.debug(f"Error parsing devices in group XML: {e}")

                                                    if result[matched_fw] is not None:
                                                        break
                                            except ET.ParseError as e:
                                                logger.debug(f"Error parsing show device groups XML: {e}")

                                    # If operational command didn't work, try different XPath patterns to find device groups
                                    if result[matched_fw] is None:
                                        device_group_paths = [
                                            "/config/device-group",
                                            "/config/shared/device-group",
                                            "/config/devices/entry[@name='localhost.localdomain']/device-group"
                                        ]

                                    for dg_path in device_group_paths:
                                        device_groups_url = f"{PANORAMA_URL}/api/?type=config&action=get&xpath={dg_path}&key={api_key}"
                                        logger.debug(f"Trying device group path: {dg_path}")

                                        async with session.get(device_groups_url, ssl=ssl_context, timeout=30) as groups_response:
                                            if groups_response.status == 200:
                                                groups_xml = await groups_response.text()
                                                logger.debug(f"Device groups XML (first 2000 chars): {groups_xml[:2000]}...")

                                                try:
                                                    groups_root = ET.fromstring(groups_xml)

                                                    # Find all device groups - try different XPath patterns
                                                    device_groups = groups_root.findall('.//device-group/entry')
                                                    if not device_groups:
                                                        device_groups = groups_root.findall('.//entry')

                                                    logger.debug(f"Found {len(device_groups)} device groups using path {dg_path}")

                                                    if len(device_groups) > 0:
                                                        for group in device_groups:
                                                            group_name = group.get('name')
                                                            if not group_name:
                                                                continue

                                                            logger.debug(f"Checking device group '{group_name}'")

                                                            # Check if this device is in this group (match by serial)
                                                            # Try different XPath patterns for devices
                                                            devices_in_group = group.findall('.//devices/entry')
                                                            if not devices_in_group:
                                                                devices_in_group = group.findall('devices/entry')
                                                            if not devices_in_group:
                                                                devices_in_group = group.findall('.//entry')

                                                            logger.debug(f"Found {len(devices_in_group)} devices in group '{group_name}'")

                                                            for dev in devices_in_group:
                                                                dev_serial = dev.get('name') or dev.text
                                                                if not dev_serial:
                                                                    # Try to get serial from nested elements
                                                                    serial_elem = dev.find('serial')
                                                                    if serial_elem is not None:
                                                                        dev_serial = serial_elem.text

                                                                logger.debug(f"Comparing device serial '{dev_serial}' with '{device_serial}'")
                                                                if dev_serial and (dev_serial == device_serial or dev_serial.lower() == device_serial.lower()):
                                                                    result[matched_fw] = group_name
                                                                    logger.debug(f"Matched firewall '{matched_fw}' (hostname: '{device_hostname}', serial: '{device_serial}') to device group '{group_name}'")
                                                                    break

                                                            # If we found a match, break out of group loop
                                                            if result[matched_fw] is not None:
                                                                break

                                                        # If we found device groups and matched, break out of path loop
                                                        if result[matched_fw] is not None:
                                                            break

                                                except ET.ParseError as e:
                                                    logger.debug(f"Error parsing device groups XML: {e}")

                                            # If we found a match, no need to try other paths
                                            if result[matched_fw] is not None:
                                                break

                                    # If still no match found after trying all methods, default to "Shared"
                                    if result[matched_fw] is None:
                                        result[matched_fw] = "Shared"
                                        logger.debug(f"No device group found for firewall '{matched_fw}', defaulting to 'Shared'")

                        if any(v is not None for v in result.values()):
                            logger.debug("Successfully retrieved device groups")
                        else:
                            logger.debug(f"No device groups found for firewalls {firewall_names}")

                    except ET.ParseError as e:
                        logger.error(f"Error parsing devices XML: {e}")
                        import traceback
                        logger.error(f"XML Parse Traceback: {traceback.format_exc()}")
                else:
                    response_text = await devices_response.text()
                    logger.error(f"Devices API call failed! HTTP {devices_response.status}: {response_text[:500]}")

    except Exception as e:
        logger.error(f"Exception querying device groups: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

    logger.debug(f"Final device groups for firewalls: {result}")
    return result
