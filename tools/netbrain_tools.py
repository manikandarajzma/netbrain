"""
NetBrain domain module – MCP tools for network path calculation and policy checking.

Exposes the following MCP tools:
  - query_network_path   – hop-by-hop path between two IPs via NetBrain
  - check_path_allowed   – allowed / denied verdict for traffic between two IPs

All NetBrain API interaction, caching, and helper logic lives here.
The shared FastMCP instance (`mcp`) is imported from tools.shared so that
tool registrations are picked up by the central server.
"""

import sys
import ssl
import json
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List

from tools.shared import mcp, NETBRAIN_URL
import netbrainauth
from tools.panorama_tools import _add_panorama_zones_to_hops, _add_panorama_device_groups_to_hops


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

# Cache for device type mappings (numeric code -> name)
_device_type_cache: Optional[Dict[int, str]] = None

# Cache for device name -> type name mappings (from Devices API)
_device_name_to_type_cache: Optional[Dict[str, str]] = None

# Debug info for Devices API call (to be included in result)
_devices_api_debug_info: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

async def get_device_type_mapping() -> Dict[int, str]:
    """
    Get device type code to name mapping from NetBrain API.
    Caches the result to avoid repeated API calls.

    Returns:
        Dictionary mapping device type codes (int) to descriptive names (str)
    """
    global _device_type_cache, _device_name_to_type_cache

    # Return cached mapping if available
    # Note: _device_type_cache might be {} (empty dict) if only name cache was built
    # So we check if name cache exists OR if numeric cache has entries
    if _device_name_to_type_cache is not None:
        # Name cache exists, return numeric cache (even if empty - name cache will be used)
        return _device_type_cache or {}
    if _device_type_cache is not None and len(_device_type_cache) > 0:
        # Numeric cache exists and has entries, return it
        return _device_type_cache

    # Get authentication token
    auth_token = netbrainauth.get_auth_token()
    if not auth_token:
        print("WARNING: Could not get auth token for device type mapping", file=sys.stderr, flush=True)
        return {}

    # Create SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    device_type_map = {}

    # Helper function to extract device types from response
    def extract_device_types_from_response(data: Any) -> List[Dict[str, Any]]:
        """Extract device type list from various response structures."""
        device_types = []
        if isinstance(data, dict):
            # Check for operationResult structure (common in SystemModel endpoints)
            if "operationResult" in data:
                op_result = data["operationResult"]
                if isinstance(op_result, dict):
                    # Check for data field in operationResult
                    if "data" in op_result:
                        device_types = op_result["data"] if isinstance(op_result["data"], list) else [op_result["data"]]
                    # Check if operationResult itself contains device types
                    elif "deviceTypes" in op_result:
                        device_types = op_result["deviceTypes"] if isinstance(op_result["deviceTypes"], list) else [op_result["deviceTypes"]]
                elif isinstance(op_result, list):
                    device_types = op_result
            elif "deviceTypes" in data:
                device_types = data["deviceTypes"] if isinstance(data["deviceTypes"], list) else [data["deviceTypes"]]
            elif "result" in data:
                result = data["result"]
                if isinstance(result, list):
                    device_types = result
                elif isinstance(result, dict) and "deviceTypes" in result:
                    device_types = result["deviceTypes"] if isinstance(result["deviceTypes"], list) else [result["deviceTypes"]]
            elif "data" in data:
                device_types = data["data"] if isinstance(data["data"], list) else [data["data"]]
            # If data itself is a dict of device types
            elif len(data) > 0 and all(isinstance(v, dict) for v in data.values() if isinstance(v, (list, dict))):
                device_types = list(data.values())
        elif isinstance(data, list):
            device_types = data
        return device_types

    # Helper function to process device types and build mapping
    def process_device_types(device_types: List[Dict[str, Any]]) -> Dict[int, str]:
        """Process device type list and build mapping dictionary."""
        mapping = {}
        for dt in device_types:
            if isinstance(dt, dict):
                # Try different field names for ID and name
                dt_id = (dt.get("id") or dt.get("ID") or dt.get("deviceTypeId") or
                        dt.get("deviceTypeID") or dt.get("typeId") or dt.get("typeID") or
                        dt.get("devType") or dt.get("devTypeId") or dt.get("deviceType"))
                dt_name = (dt.get("deviceType") or dt.get("DeviceType") or dt.get("name") or
                          dt.get("Name") or dt.get("typeName") or dt.get("TypeName") or
                          dt.get("description") or dt.get("Description") or dt.get("displayName") or
                          dt.get("DisplayName") or dt.get("subTypeName") or dt.get("SubTypeName"))

                if dt_id and dt_name:
                    try:
                        mapping[int(dt_id)] = str(dt_name)
                    except (ValueError, TypeError):
                        pass
        return mapping

    try:
        # Try common NetBrain API endpoints for device types
        # The correct endpoint is /ServicesAPI/SystemModel/getAllDisplayDeviceTypes
        # SystemModel endpoints may require Bearer token authentication, but Token header also works
        api_endpoints = [
            (f"{NETBRAIN_URL}/ServicesAPI/SystemModel/getAllDisplayDeviceTypes", False),  # Try Token header first
            (f"{NETBRAIN_URL}/ServicesAPI/SystemModel/getAllDisplayDeviceTypes", True),   # Fallback to Bearer
            (f"{NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/DeviceType", False),  # Token header
            (f"{NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/DeviceTypes", False),  # Token header
            (f"{NETBRAIN_URL}/ServicesAPI/API/V1/Admin/DeviceType", False),  # Token header
            (f"{NETBRAIN_URL}/ServicesAPI/API/V1/Admin/DeviceTypes", False),  # Token header
        ]

        # Also try fetching device types from devices API as fallback
        devices_endpoint = f"{NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Devices"

        async with aiohttp.ClientSession() as session:
            for endpoint, use_bearer in api_endpoints:
                try:
                    # Use Bearer token for SystemModel endpoints, Token header for others
                    if use_bearer:
                        headers = {
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                            "Authorization": f"Bearer {auth_token}"
                        }
                    else:
                        headers = {
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                            "Token": auth_token
                        }

                    print(f"DEBUG: Trying device type endpoint: {endpoint} (Bearer={use_bearer})", file=sys.stderr, flush=True)
                    async with session.get(endpoint, headers=headers, ssl=ssl_context, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            print(f"DEBUG: Device type API response: {json.dumps(data, indent=2)[:500]}...", file=sys.stderr, flush=True)

                            # Extract device types from response
                            device_types = extract_device_types_from_response(data)

                            # Process device types and build mapping
                            device_type_map = process_device_types(device_types)

                            if device_type_map:
                                print(f"DEBUG: Successfully loaded {len(device_type_map)} device type mappings", file=sys.stderr, flush=True)
                                _device_type_cache = device_type_map
                                return device_type_map
                        elif response.status == 401:
                            # If Bearer failed, try Token header (for SystemModel endpoints)
                            if use_bearer:
                                print(f"DEBUG: Bearer token failed, trying Token header...", file=sys.stderr, flush=True)
                                token_headers = {
                                    "Content-Type": "application/json",
                                    "Accept": "application/json",
                                    "Token": auth_token
                                }
                                async with session.get(endpoint, headers=token_headers, ssl=ssl_context, timeout=10) as token_response:
                                    if token_response.status == 200:
                                        data = await token_response.json()
                                        device_types = extract_device_types_from_response(data)
                                        device_type_map = process_device_types(device_types)
                                        if device_type_map:
                                            print(f"DEBUG: Successfully loaded {len(device_type_map)} device type mappings with Token header", file=sys.stderr, flush=True)
                                            _device_type_cache = device_type_map
                                            return device_type_map
                            error_text = await response.text()
                            print(f"DEBUG: Authentication failed for {endpoint}: {error_text[:200]}", file=sys.stderr, flush=True)
                            continue
                        elif response.status == 404:
                            # Endpoint doesn't exist, try next one
                            continue
                        else:
                            error_text = await response.text()
                            print(f"DEBUG: Device type API returned {response.status}: {error_text[:200]}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"DEBUG: Error querying device type endpoint {endpoint}: {e}", file=sys.stderr, flush=True)
                    continue

        # If no endpoint worked, try fetching from devices API as fallback
        # Note: Devices API doesn't provide numeric deviceType IDs, only subTypeName
        # We'll build a name-based lookup cache that can be used when processing path hops
        print("DEBUG: Direct device type endpoints failed, trying devices API as fallback...", file=sys.stderr, flush=True)

        # Store debug info to return in result
        devices_api_debug = {
            "attempted": True,
            "endpoint": devices_endpoint,
            "status": None,
            "error": None,
            "devices_count": 0,
            "cache_built": False
        }

        try:
            token_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Token": auth_token
            }

            # Fetch a sample of devices to extract device types
            # Note: API requires limit between 10 and 100
            params = {
                "version": 1,
                "skip": 0,
                "limit": 100  # Maximum allowed by API
            }

            print(f"DEBUG: Calling Devices API: {devices_endpoint} with params: {params}", file=sys.stderr, flush=True)
            async with aiohttp.ClientSession() as session:
                async with session.get(devices_endpoint, headers=token_headers, params=params, ssl=ssl_context, timeout=30) as response:
                    devices_api_debug["status"] = response.status
                    print(f"DEBUG: Devices API response status: {response.status}", file=sys.stderr, flush=True)
                    if response.status == 200:
                        data = await response.json()
                        print(f"DEBUG: Devices API response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}", file=sys.stderr, flush=True)
                        print(f"DEBUG: Devices API statusCode: {data.get('statusCode') if isinstance(data, dict) else 'N/A'}", file=sys.stderr, flush=True)

                        if data.get("statusCode") == 790200:
                            devices = data.get("devices", [])
                            devices_api_debug["devices_count"] = len(devices)
                            print(f"DEBUG: Fetched {len(devices)} devices from devices API", file=sys.stderr, flush=True)

                            if len(devices) == 0:
                                print(f"DEBUG: WARNING: Devices API returned 0 devices!", file=sys.stderr, flush=True)
                                devices_api_debug["error"] = "Devices API returned 0 devices"
                                # Try without limit parameter
                                print(f"DEBUG: Retrying Devices API without limit parameter...", file=sys.stderr, flush=True)
                                async with session.get(devices_endpoint, headers=token_headers, ssl=ssl_context, timeout=30) as retry_response:
                                    devices_api_debug["retry_status"] = retry_response.status
                                    if retry_response.status == 200:
                                        retry_data = await retry_response.json()
                                        if retry_data.get("statusCode") == 790200:
                                            devices = retry_data.get("devices", [])
                                            devices_api_debug["devices_count"] = len(devices)
                                            print(f"DEBUG: Retry fetched {len(devices)} devices", file=sys.stderr, flush=True)

                            # Extract device type mappings from devices
                            # Since Devices API doesn't provide numeric IDs, we'll try to find them
                            # or build a name-based cache for later lookup
                            device_name_to_type = {}  # Cache for device name -> type name

                            for dev in devices:
                                dev_name = dev.get("name") or dev.get("hostname") or dev.get("mgmtIP")
                                dev_type_name = dev.get("subTypeName") or dev.get("deviceTypeName") or dev.get("typeName")

                                # Debug: Print first few devices to see structure
                                if len(device_name_to_type) < 3:
                                    print(f"DEBUG: Sample device - name: '{dev_name}', subTypeName: '{dev_type_name}', keys: {list(dev.keys())[:10]}", file=sys.stderr, flush=True)

                                # Try to find numeric device type ID
                                dev_type_id = dev.get("deviceType") or dev.get("devType") or dev.get("typeId")

                                if dev_type_id and dev_type_name:
                                    try:
                                        device_type_map[int(dev_type_id)] = str(dev_type_name)
                                    except (ValueError, TypeError):
                                        pass

                                # Also build name-based cache for fallback
                                if dev_name and dev_type_name:
                                    device_name_to_type[str(dev_name)] = str(dev_type_name)

                            print(f"DEBUG: Built name-based cache with {len(device_name_to_type)} entries", file=sys.stderr, flush=True)
                            if len(device_name_to_type) > 0:
                                print(f"DEBUG: Sample cache entries: {list(device_name_to_type.items())[:5]}", file=sys.stderr, flush=True)

                            # Store name-based cache as a module-level variable for later use
                            _device_name_to_type_cache = device_name_to_type
                            devices_api_debug["cache_built"] = len(device_name_to_type) > 0
                            devices_api_debug["cache_size"] = len(device_name_to_type)

                            if device_type_map:
                                print(f"DEBUG: Successfully loaded {len(device_type_map)} device type mappings from devices API", file=sys.stderr, flush=True)
                                _device_type_cache = device_type_map
                                return device_type_map
                            elif device_name_to_type:
                                print(f"DEBUG: Built name-based device type cache with {len(device_name_to_type)} entries (no numeric IDs found)", file=sys.stderr, flush=True)
                                print(f"DEBUG: Name cache sample: {list(device_name_to_type.keys())[:10]}", file=sys.stderr, flush=True)
                                # Return empty dict but cache is available for name-based lookup
                                _device_type_cache = {}  # Cache empty dict to avoid repeated failed attempts
                                return {}
                            else:
                                print(f"DEBUG: WARNING: No devices found or no device types extracted!", file=sys.stderr, flush=True)
                                devices_api_debug["error"] = "No device types extracted from devices"
                        else:
                            status_desc = data.get('statusDescription', 'No description') if isinstance(data, dict) else 'Unknown'
                            print(f"DEBUG: Devices API returned statusCode {data.get('statusCode')}: {status_desc}", file=sys.stderr, flush=True)
                            devices_api_debug["error"] = f"statusCode {data.get('statusCode')}: {status_desc}"
                    else:
                        error_text = await response.text()
                        print(f"DEBUG: Devices API returned HTTP {response.status}: {error_text[:500]}", file=sys.stderr, flush=True)
                        devices_api_debug["error"] = f"HTTP {response.status}: {error_text[:200]}"
        except Exception as e:
            print(f"DEBUG: Error fetching from devices API: {e}", file=sys.stderr, flush=True)
            import traceback
            print(f"DEBUG: Devices API error traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
            devices_api_debug["error"] = str(e)

        # Store debug info in module-level variable so it can be added to result
        global _devices_api_debug_info
        _devices_api_debug_info = devices_api_debug

        # If no endpoint worked, return empty dict
        print("WARNING: Could not retrieve device type mappings from NetBrain API", file=sys.stderr, flush=True)
        _device_type_cache = {}  # Cache empty dict to avoid repeated failed attempts
        return {}

    except Exception as e:
        print(f"ERROR: Exception getting device type mapping: {str(e)}", file=sys.stderr, flush=True)
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        _device_type_cache = {}  # Cache empty dict
        return {}


def map_device_type(dev_type: Any, device_type_map: Optional[Dict[int, str]] = None, device_name: Optional[str] = None) -> str:
    """
    Map NetBrain device type code to descriptive name.
    Uses device name lookup as primary method (from Devices API cache),
    falls back to numeric code mapping if name not available.

    Args:
        dev_type: Device type code (number or string)
        device_type_map: Optional device type mapping dictionary (if None, uses cached mapping)
        device_name: Optional device name to look up in Devices API cache

    Returns:
        Descriptive device type name or original code if not found
    """
    # First, try to look up by device name (preferred method - uses subTypeName from Devices API)
    if device_name:
        global _device_name_to_type_cache
        # Force cache rebuild if it doesn't exist (synchronous call - this is a sync function)
        if _device_name_to_type_cache is None:
            print(f"DEBUG: map_device_type - Name cache is None, attempting to build it synchronously...", file=sys.stderr, flush=True)
            # We can't await here, but we can trigger an async task
            # For now, just log and continue - the cache should be built by get_device_type_mapping()
            print(f"DEBUG: map_device_type - WARNING: Name cache not built yet, device_name='{device_name}', dev_type='{dev_type}'", file=sys.stderr, flush=True)

        if _device_name_to_type_cache:
            # Try exact match first
            device_type_name = _device_name_to_type_cache.get(device_name)
            if device_type_name:
                print(f"DEBUG: map_device_type - Found '{device_name}' -> '{device_type_name}' in name cache (exact match)", file=sys.stderr, flush=True)
                return device_type_name

            # Try case-insensitive match
            device_name_lower = device_name.lower()
            for cached_name, cached_type in _device_name_to_type_cache.items():
                if cached_name.lower() == device_name_lower:
                    print(f"DEBUG: map_device_type - Found '{device_name}' -> '{cached_type}' in name cache (case-insensitive match, cached as '{cached_name}')", file=sys.stderr, flush=True)
                    return cached_type

            print(f"DEBUG: map_device_type - Device '{device_name}' not found in name cache (cache has {len(_device_name_to_type_cache)} entries: {list(_device_name_to_type_cache.keys())[:10]}...)", file=sys.stderr, flush=True)
        else:
            print(f"DEBUG: map_device_type - Name cache is empty dict, device_name='{device_name}', dev_type='{dev_type}'", file=sys.stderr, flush=True)

    # Fallback to numeric code mapping
    if dev_type is None:
        return ""

    # Convert to string and handle both string and numeric types
    dev_type_str = str(dev_type).strip()
    if not dev_type_str or dev_type_str == "None":
        return ""

    # Try to convert to int for numeric comparison
    try:
        dev_type_num = int(dev_type_str)
    except (ValueError, TypeError):
        # If not numeric, return as-is (might already be a descriptive name)
        return dev_type_str

    # Use provided map or get from cache
    if device_type_map is None:
        # This will use cached mapping if available
        device_type_map = _device_type_cache or {}

    # Return mapped name or original code if not found
    return device_type_map.get(dev_type_num, f"Device Type {dev_type_num}")


async def _query_network_path_impl(
    source: str,
    destination: str,
    protocol: str,
    port: str,
    is_live: int = 1,
    continue_on_policy_denial: bool = True
):
    """Internal path calculation implementation. Used by query_network_path and check_path_allowed."""
    # Declare globals at the top of the function
    global _device_name_to_type_cache, _devices_api_debug_info

    # Import json at the top of the function to avoid scoping issues
    import json

    # Debug: Print function entry and parameters
    print(f"DEBUG: _query_network_path_impl called with source={source}, destination={destination}, protocol={protocol}, port={port}, is_live={is_live}, continue_on_policy_denial={continue_on_policy_denial}", file=sys.stderr, flush=True)

    # Get authentication token from netbrainauth module
    # This token is required for all NetBrain API requests
    auth_token = netbrainauth.get_auth_token()

    # Check if authentication token was successfully obtained
    # If not, return an error dictionary immediately
    if not auth_token:
        print("DEBUG: Failed to get authentication token", file=sys.stderr, flush=True)
        return {"error": "Failed to get authentication token"}

    print(f"DEBUG: Authentication token obtained: {auth_token[:20]}...", file=sys.stderr, flush=True)

    # Pre-build device type cache to ensure it's available when processing path hops
    print(f"DEBUG: Pre-building device type cache...", file=sys.stderr, flush=True)
    await get_device_type_mapping()
    if _device_name_to_type_cache:
        print(f"DEBUG: Device type cache ready with {len(_device_name_to_type_cache)} name-based entries", file=sys.stderr, flush=True)
    else:
        print(f"DEBUG: WARNING: Device type name cache is not available", file=sys.stderr, flush=True)

    # Helper: build headers with the given token
    def _make_headers(token):
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Token": token
        }

    # Prepare HTTP headers for all API requests
    # NetBrain API uses "Token" header (capital T) for authentication
    headers = _make_headers(auth_token)

    # Map protocol string to protocol number
    # Protocol numbers: 4=IPv4, 6=TCP, 17=UDP
    protocol_map = {
        "TCP": 6,  # TCP protocol number
        "UDP": 17,  # UDP protocol number
        "IP": 4,  # IPv4 protocol number
        "IPv4": 4  # IPv4 protocol number
    }
    protocol_num = protocol_map.get(protocol.upper(), 4)  # Default to IPv4 if not found

    # Convert port to integer, default to 0 if not provided or invalid
    try:
        source_port = int(port) if port else 0
        dest_port = int(port) if port else 0
    except ValueError:
        source_port = 0
        dest_port = 0

    # Trim source and destination to remove whitespace
    source_trimmed = source.strip() if source else ""
    destination_trimmed = destination.strip() if destination else ""

    # Validate source is not empty
    if not source_trimmed:
        print("ERROR: Source IP/hostname is empty", file=sys.stderr, flush=True)
        return {
            "error": "Source IP/hostname cannot be empty",
            "step": 0
        }

    # Create SSL context for HTTPS connections
    # Disable SSL verification for self-signed certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Create an async HTTP client session with timeout
    timeout = aiohttp.ClientTimeout(total=60, connect=30, sock_read=30)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # ====================================================================
            # STEP 1: Resolve Device Gateway
            # GET /V1/CMDB/Path/Gateways?ipOrHost=<source>
            # ====================================================================
            print(f"DEBUG: Step 1 - Resolving gateway for source: {source_trimmed}", file=sys.stderr, flush=True)
            gateway_url = f"{NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Path/Gateways"
            gateway_params = {"ipOrHost": source_trimmed}

            # Debug: Print full request details
            import urllib.parse
            full_url_with_params = f"{gateway_url}?{urllib.parse.urlencode(gateway_params)}"
            print(f"DEBUG: Step 1 - Full URL: {full_url_with_params}", file=sys.stderr, flush=True)
            print(f"DEBUG: Step 1 - Headers: {headers}", file=sys.stderr, flush=True)
            print(f"DEBUG: Step 1 - Params: {gateway_params}", file=sys.stderr, flush=True)

            # Gateway resolution with retry-on-401 (expired token).
            # Try up to 2 times: if the first attempt gets HTTP 401, refresh the
            # token and retry once before giving up.
            source_gateway = None
            for _gw_attempt in range(2):
              async with session.get(gateway_url, headers=headers, params=gateway_params, ssl=ssl_context) as gateway_response:
                response_status = gateway_response.status
                print(f"DEBUG: Step 1 (attempt {_gw_attempt+1}) - Response status: {response_status}", file=sys.stderr, flush=True)

                if response_status == 401 and _gw_attempt == 0:
                    # Token expired - refresh and retry
                    print(f"WARNING: Step 1 - HTTP 401 (token expired), refreshing token and retrying...", file=sys.stderr, flush=True)
                    netbrainauth.clear_token_cache()
                    auth_token = netbrainauth.get_auth_token()
                    if not auth_token:
                        return {
                            "error": "Authentication failed: could not refresh expired token. Check NetBrain credentials.",
                            "step": 1,
                            "source": source_trimmed
                        }
                    headers = _make_headers(auth_token)
                    print(f"DEBUG: Step 1 - New token obtained: {auth_token[:20]}..., retrying gateway resolution", file=sys.stderr, flush=True)
                    continue  # retry the for loop

                if response_status != 200:
                    # Read response body as text first
                    error_text = await gateway_response.text()
                    print(f"DEBUG: Step 1 - Error response text: {error_text[:500]}", file=sys.stderr, flush=True)

                    # Try to parse as JSON for more details
                    error_details = error_text
                    status_code = None
                    status_desc = None
                    try:
                        error_json = json.loads(error_text)
                        if isinstance(error_json, dict):
                            status_code = error_json.get("statusCode", "Unknown")
                            status_desc = error_json.get("statusDescription", "No description")
                            error_details = f"statusCode: {status_code}, statusDescription: {status_desc}"
                            print(f"WARNING: Step 1 - HTTP {response_status}, {error_details}", file=sys.stderr, flush=True)
                        else:
                            error_details = str(error_json)
                            print(f"WARNING: Step 1 - HTTP {response_status}: {error_details}", file=sys.stderr, flush=True)
                    except:
                        # If JSON parsing fails, use text as-is
                        error_details = error_text
                        print(f"WARNING: Step 1 - HTTP {response_status}: {error_details}", file=sys.stderr, flush=True)

                    # If gateway was not found (792040) and fix-up rules are enabled, proceed anyway
                    # The fix-up rules will handle gateway resolution during path calculation
                    if status_code == 792040:  # Gateway was not found
                        print(f"INFO: Gateway not found (statusCode 792040), but fix-up rules are enabled. Proceeding to path calculation...", file=sys.stderr, flush=True)
                        # Create a minimal valid gateway object - fix-up rules will override this
                        # Use source IP as placeholder, fix-up rules will replace it
                        source_gateway = {
                            "gatewayName": source_trimmed  # Use source IP as placeholder, fix-up rules will replace it
                            # Omit type and payload - let fix-up rules handle it
                        }
                        print(f"DEBUG: Step 1 - Using placeholder gateway ({source_trimmed}), fix-up rules will apply during path calculation", file=sys.stderr, flush=True)
                    else:
                        # For other errors, return error
                        return {
                            "error": f"Failed to resolve gateway: HTTP {response_status}",
                            "step": 1,
                            "details": error_details,
                            "source": source_trimmed,
                            "gateway_url": gateway_url,
                            "gateway_params": gateway_params
                        }
                else:
                    # HTTP 200 - read JSON response
                    gateway_data = await gateway_response.json()

                    # Check if gateway resolution was successful
                    if gateway_data.get("statusCode") != 790200:
                        status_code = gateway_data.get("statusCode", "Unknown")
                        status_desc = gateway_data.get("statusDescription", "No description")
                        print(f"WARNING: Step 1 - Gateway resolution failed - statusCode={status_code}: {status_desc}", file=sys.stderr, flush=True)

                        # If gateway was not found (792040) and fix-up rules are enabled, proceed anyway
                        # The fix-up rules will handle gateway resolution during path calculation
                        if status_code == 792040:  # Gateway was not found
                            print(f"INFO: Gateway not found, but fix-up rules are enabled. Proceeding to path calculation...", file=sys.stderr, flush=True)
                            # Create a minimal valid gateway object - fix-up rules will override this
                            # Use source IP as placeholder, fix-up rules will replace it
                            source_gateway = {
                                "gatewayName": source_trimmed,  # Use source IP as placeholder
                                "type": "",
                                "payload": None
                            }
                            print(f"DEBUG: Step 1 - Using placeholder gateway ({source_trimmed}), fix-up rules will apply during path calculation", file=sys.stderr, flush=True)
                        else:
                            # For other errors, return error
                            return {
                                "error": f"Gateway resolution failed: statusCode={status_code}",
                                "step": 1,
                                "statusDescription": status_desc,
                                "response": gateway_data
                            }
                    else:
                        # Get the gateway list from the response
                        gateway_list = gateway_data.get("gatewayList", [])
                        if not gateway_list:
                            print("WARNING: Step 1 - No gateways found in response, but fix-up rules are enabled. Proceeding...", file=sys.stderr, flush=True)
                            # Create a minimal valid gateway object - fix-up rules will override this
                            # Use source IP as placeholder, fix-up rules will replace it
                            source_gateway = {
                                "gatewayName": source_trimmed,  # Use source IP as placeholder
                                "type": "",
                                "payload": None
                            }
                            print(f"DEBUG: Step 1 - Using placeholder gateway ({source_trimmed}), fix-up rules will apply during path calculation", file=sys.stderr, flush=True)
                        else:
                            # Use the first gateway from the list
                            # The gateway object contains: gatewayName, type, payload
                            source_gateway = gateway_list[0]
                            print(f"DEBUG: Step 1 - Gateway resolved: {source_gateway.get('gatewayName', 'Unknown')}", file=sys.stderr, flush=True)
                            print(f"DEBUG: Step 1 - Gateway object structure: {json.dumps(source_gateway, indent=2)}", file=sys.stderr, flush=True)

                # Break out of retry loop after processing (only 'continue' on 401 should loop)
                break

            # Proceed to Step 2 only if we have a gateway (either resolved or null for fix-up rules)
            if source_gateway is None:
                return {
                    "error": "Failed to resolve gateway and no fix-up rule fallback available",
                    "step": 1,
                    "source": source_trimmed
                }

            # ====================================================================
            # STEP 2: Calculate Path
            # POST /V1/CMDB/Path/Calculation
            # ====================================================================
            print(f"DEBUG: Step 2 - Calculating path from {source_trimmed} to {destination_trimmed}", file=sys.stderr, flush=True)
            calc_url = f"{NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Path/Calculation"

            # Build payload for path calculation
            # sourceGateway must be the object from Step 1, not separate fields
            # However, if gateway is null (all None), omit it to let fix-up rules apply
            payload = {
                "sourceIP": source_trimmed,  # IP address of the source device
                "sourcePort": source_port,  # Source port (0 if not provided)
                "destIP": destination_trimmed,  # IP address of the destination device
                "destPort": dest_port,  # Destination port (0 if not provided)
                "pathAnalysisSet": 1,  # 1=L3 Path; 2=L2 Path; 3=L3 Active Path
                "protocol": protocol_num,  # Protocol number (4=IPv4, 6=TCP, 17=UDP)
                "isLive": 1 if is_live else 0,  # 0=Current Baseline; 1=Live access
                "advanced": {
                    "advanced.debugMode": True,
                    "calcWhenDeniedByACL": True,
                    "calcWhenDeniedByPolicy": continue_on_policy_denial,  # Continue calculation even if denied by device-level or subnet-level policy
                    "enablePathFixup": True,
                    "enablePathIPAndGatewayFixup": True  # Enable Path IP and Gateway Fix-up Rule
                }
            }

            # Always include sourceGateway - it's required by the API
            # If gateway resolution failed, we use a placeholder that fix-up rules will override
            payload["sourceGateway"] = source_gateway
            print(f"DEBUG: Step 2 - Including sourceGateway: {source_gateway.get('gatewayName', 'Unknown')}", file=sys.stderr, flush=True)

            print(f"DEBUG: Step 2 - Payload: {json.dumps(payload, indent=2)}", file=sys.stderr, flush=True)

            async with session.post(calc_url, headers=headers, json=payload, ssl=ssl_context) as calc_response:
                if calc_response.status != 200:
                    error_text = await calc_response.text()
                    print(f"ERROR: Step 2 failed - HTTP {calc_response.status}: {error_text}", file=sys.stderr, flush=True)
                    return {
                        "error": f"Path calculation failed: HTTP {calc_response.status}",
                        "step": 2,
                        "details": error_text,
                        "payload_sent": payload,
                        "auth_token": auth_token
                    }

                calc_data = await calc_response.json()

                # Check if path calculation was successful
                if calc_data.get("statusCode") != 790200:
                    status_code = calc_data.get("statusCode", "Unknown")
                    status_desc = calc_data.get("statusDescription", "No description")
                    print(f"ERROR: Step 2 failed - statusCode={status_code}: {status_desc}", file=sys.stderr, flush=True)
                    return {
                        "error": f"Path calculation failed: statusCode={status_code}",
                        "step": 2,
                        "statusDescription": status_desc,
                        "response": calc_data,
                        "payload_sent": payload,
                        "auth_token": auth_token
                    }

                # Get taskID from the response
                # Ensure taskID is a string (JSON may return it as different type)
                task_id = calc_data.get("taskID")
                if not task_id:
                    print("ERROR: Step 2 - No taskID in response", file=sys.stderr, flush=True)
                    return {
                        "error": "No taskID returned from path calculation",
                        "step": 2,
                        "response": calc_data
                    }

                # Convert taskID to string to ensure it's in the correct format
                task_id = str(task_id)

                print(f"DEBUG: Step 2 - Path calculation successful, taskID: {task_id}", file=sys.stderr, flush=True)

            # ====================================================================
            # STEP 3: Get Path Details (Optional)
            # GET /V1/CMDB/Path/Calculation/{taskID}/OverView
            # Note: This step is optional - path calculation (Step 2) is the main operation
            # Path details may not be immediately available or may require different endpoint
            # The taskID from Step 2 can be used to query path details separately if needed
            # ====================================================================
            print(f"DEBUG: Step 3 - Attempting to get path details for taskID: {task_id} (optional)", file=sys.stderr, flush=True)

            path_data = None
            step3_error = None

            # Try to get path details, but don't fail if this step doesn't work
            # Some NetBrain versions may have different endpoint formats or timing requirements
            # Path calculation is asynchronous, especially for live data, so we need to poll
            try:
                # Primary endpoint from documentation
                path_url = f"{NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Path/Calculation/{task_id}/OverView"

                # Poll for path details - live data calculations can take longer
                # Use longer polling for live data (is_live=1) since it takes more time
                if is_live:
                    max_attempts = 120  # Maximum number of polling attempts (4 minutes for live data)
                    initial_poll_interval = 2  # Start with 2 seconds
                    max_poll_interval = 5  # Increase to 5 seconds after many attempts
                else:
                    max_attempts = 30  # Maximum number of polling attempts (60 seconds for baseline)
                    initial_poll_interval = 2
                    max_poll_interval = 3

                max_wait_time = max_attempts * max_poll_interval  # Approximate max wait time

                print(f"DEBUG: Step 3 - Polling endpoint: {path_url} (max {max_attempts} attempts, {initial_poll_interval}-{max_poll_interval}s interval)", file=sys.stderr, flush=True)

                for attempt in range(1, max_attempts + 1):
                    # Wait before each attempt (except the first one)
                    # Use progressive backoff: increase interval after 20 attempts
                    if attempt > 1:
                        if attempt <= 20:
                            poll_interval = initial_poll_interval
                        else:
                            # Progressive backoff: increase interval gradually
                            poll_interval = min(initial_poll_interval + (attempt - 20) // 10, max_poll_interval)
                        await asyncio.sleep(poll_interval)

                    print(f"DEBUG: Step 3 - Attempt {attempt}/{max_attempts}", file=sys.stderr, flush=True)

                    async with session.get(path_url, headers=headers, ssl=ssl_context) as path_response:
                        # Read the response text first (can only read once)
                        response_text = await path_response.text()

                        # Try to parse as JSON regardless of status code
                        try:
                            response_json = json.loads(response_text)

                            # Check if task is not finished yet (statusCode 794007)
                            status_code = response_json.get("statusCode")
                            if status_code == 794007:
                                status_desc = response_json.get("statusDescription", "")
                                print(f"DEBUG: Step 3 - Attempt {attempt}: Task not finished yet: {status_desc}", file=sys.stderr, flush=True)
                                # Continue polling
                                continue

                            # Check if response contains path data (path_overview, path_list, etc.)
                            if "path_overview" in response_json or "path_list" in response_json or "hop_detail_list" in response_json:
                                # Response contains path data, use it even if status is not 200
                                path_data = response_json
                                print(f"DEBUG: Step 3 - Path details retrieved on attempt {attempt} (status {path_response.status})", file=sys.stderr, flush=True)

                                # If status is not 200, note it but still use the data
                                if path_response.status != 200:
                                    print(f"INFO: Step 3 - Path details available but HTTP status is {path_response.status}", file=sys.stderr, flush=True)
                                    # Store the status code in the data for reference
                                    path_data["_http_status"] = path_response.status
                                # Success - break out of polling loop
                                break
                            elif path_response.status == 200 and status_code == 790200:
                                # Standard success case
                                path_data = response_json
                                print(f"DEBUG: Step 3 - Path details retrieved successfully on attempt {attempt}", file=sys.stderr, flush=True)
                                # Success - break out of polling loop
                                break
                            else:
                                # Check if this is a final error (not "not finished yet")
                                if status_code != 794007:
                                    # No path data in response and it's not a "not finished" error
                                    print(f"DEBUG: Step 3 - HTTP {path_response.status}, statusCode {status_code}: {response_text[:200]}", file=sys.stderr, flush=True)
                                    step3_error = f"HTTP {path_response.status}: {response_text[:200]}"
                                    # Don't continue polling if we got a different error
                                    break
                                # Otherwise continue polling
                        except json.JSONDecodeError:
                            # Response is not valid JSON
                            if path_response.status == 200:
                                # Try to use it anyway
                                path_data = {"raw_response": response_text}
                                break
                            else:
                                print(f"DEBUG: Step 3 - HTTP {path_response.status}: Invalid JSON response", file=sys.stderr, flush=True)
                                step3_error = f"HTTP {path_response.status}: {response_text[:200]}"
                                break

                    # If we got path_data, break out of the loop
                    if path_data is not None:
                        break

                # If we exhausted all attempts without getting data
                if path_data is None and step3_error is None:
                    step3_error = f"Task did not complete within {max_wait_time} seconds after {max_attempts} polling attempts"
                    print(f"DEBUG: Step 3 - {step3_error}", file=sys.stderr, flush=True)

            except Exception as e:
                print(f"DEBUG: Step 3 - Exception during path details retrieval: {str(e)}", file=sys.stderr, flush=True)
                step3_error = str(e)

            # Step 3 is optional - if it fails, we still return success from Step 2
            # The taskID can be used to query path details separately
            if path_data is None:
                print(f"INFO: Step 3 - Path details not available, but path calculation (Step 2) succeeded", file=sys.stderr, flush=True)

            # Build result dictionary
            # Don't include full calc_data in path_info to avoid huge responses
            # Only include essential fields
            result = {
                "source": source_trimmed,
                "destination": destination_trimmed,
                "protocol": protocol,
                "port": port,
                "taskID": task_id,
                "statusCode": calc_data.get("statusCode"),
                "statusDescription": calc_data.get("statusDescription"),
                "gateway_used": source_gateway.get("gatewayName")
                # Removed "path_info": calc_data to avoid huge responses
            }

            # Helper function to extract path hops from various response structures
            async def extract_path_hops(data_source, source_name="response"):
                """Extract path hops from response data, handling different structures"""
                # Get device type mapping once for this extraction
                # This will build the name-based cache if it doesn't exist
                device_type_map = await get_device_type_mapping()

                # Ensure name cache is available (it might have been built even if numeric map is empty)
                global _device_name_to_type_cache
                if _device_name_to_type_cache:
                    print(f"DEBUG: extract_path_hops - Name cache available with {len(_device_name_to_type_cache)} entries: {list(_device_name_to_type_cache.keys())}", file=sys.stderr, flush=True)
                else:
                    print(f"DEBUG: extract_path_hops - Name cache is None/empty, attempting to build it...", file=sys.stderr, flush=True)
                    # Force rebuild the cache if it doesn't exist
                    device_type_map = await get_device_type_mapping()
                    if _device_name_to_type_cache:
                        print(f"DEBUG: extract_path_hops - Name cache now available with {len(_device_name_to_type_cache)} entries", file=sys.stderr, flush=True)
                    else:
                        print(f"DEBUG: extract_path_hops - WARNING: Name cache still not available after rebuild attempt", file=sys.stderr, flush=True)

                simplified_hops = []
                path_status_overall = "Unknown"
                path_failure_reason = None

                try:
                    # Try different possible response structures
                    path_overview = None

                    # Structure 1: path_overview is a list
                    if "path_overview" in data_source:
                        path_overview = data_source.get("path_overview", [])
                        if not isinstance(path_overview, list):
                            path_overview = [path_overview]

                    # Structure 2: path_overview might be directly in the response (not nested)
                    elif isinstance(data_source, list):
                        path_overview = data_source

                    # Structure 3: Check if calc_data has path information directly
                    elif "path_list" in data_source:
                        path_overview = [{"path_list": data_source.get("path_list", [])}]

                    if path_overview:
                        print(f"DEBUG: Found path_overview in {source_name}, processing {len(path_overview)} path group(s)", file=sys.stderr, flush=True)
                        for path_group in path_overview:
                            # Handle both dict and list structures
                            if isinstance(path_group, dict):
                                path_list = path_group.get("path_list", [])
                            elif isinstance(path_group, list):
                                path_list = path_group
                            else:
                                continue

                            if not isinstance(path_list, list):
                                path_list = [path_list]

                            for path in path_list:
                                if not isinstance(path, dict):
                                    continue

                                # Get path-level status and description
                                path_status_overall = path.get("status", "Unknown")
                                path_description = path.get("description", "")

                                branch_list = path.get("branch_list", [])
                                if not isinstance(branch_list, list):
                                    branch_list = [branch_list] if branch_list else []

                                for branch in branch_list:
                                    if not isinstance(branch, dict):
                                        continue

                                    branch_status = branch.get("status", "Unknown")
                                    branch_failure_reason = branch.get("failure_reason", None)

                                    hop_detail_list = branch.get("hop_detail_list", [])
                                    if not isinstance(hop_detail_list, list):
                                        hop_detail_list = [hop_detail_list] if hop_detail_list else []

                                    for hop in hop_detail_list:
                                        if not isinstance(hop, dict):
                                            continue

                                        from_dev = hop.get("fromDev", {})
                                        to_dev = hop.get("toDev", {})

                                        if not isinstance(from_dev, dict):
                                            from_dev = {}
                                        if not isinstance(to_dev, dict):
                                            to_dev = {}

                                        def _device_display_name(dev: dict) -> Optional[str]:
                                            """NetBrain has no unknown devices; use first available name field."""
                                            if not dev:
                                                return None
                                            name = (
                                                dev.get("devName") or dev.get("displayName") or dev.get("name")
                                                or dev.get("hostName") or dev.get("hostname") or dev.get("deviceName")
                                            )
                                            if name and str(name).strip():
                                                return str(name).strip()
                                            ip = dev.get("ip") or dev.get("ipAddress") or dev.get("IP")
                                            if ip:
                                                return str(ip).strip()
                                            return None

                                        from_dev_name = _device_display_name(from_dev) or "Unknown"
                                        to_dev_name = _device_display_name(to_dev)

                                        # Check if device is a firewall by examining device type and name
                                        from_dev_type = str(from_dev.get("devType", "")).lower() if isinstance(from_dev, dict) else ""
                                        to_dev_type = str(to_dev.get("devType", "")).lower() if isinstance(to_dev, dict) else ""

                                        # Debug: Print all device info for troubleshooting
                                        print(f"DEBUG: Hop device check - from: '{from_dev_name}' (type: '{from_dev_type}'), to: '{to_dev_name}' (type: '{to_dev_type}')", file=sys.stderr, flush=True)

                                        # Check if from_device is a firewall
                                        is_from_firewall = (
                                            "firewall" in from_dev_type or
                                            "fw" in from_dev_type or
                                            "fw" in from_dev_name.lower() or  # Check device name for "fw"
                                            "palo" in from_dev_name.lower() or
                                            "fortinet" in from_dev_name.lower() or
                                            "checkpoint" in from_dev_name.lower() or
                                            "asa" in from_dev_name.lower()
                                        )

                                        # Check if to_device is a firewall
                                        is_to_firewall = (
                                            to_dev_name and (
                                                "firewall" in to_dev_type or
                                                "fw" in to_dev_type or
                                                "fw" in to_dev_name.lower() or  # Check device name for "fw"
                                                "palo" in to_dev_name.lower() or
                                                "fortinet" in to_dev_name.lower() or
                                                "checkpoint" in to_dev_name.lower() or
                                                "asa" in to_dev_name.lower()
                                            )
                                        )

                                        # Debug firewall detection
                                        if is_from_firewall or is_to_firewall:
                                            print(f"DEBUG: ✓ Firewall detected - from: {from_dev_name} (type: {from_dev_type}, is_fw: {is_from_firewall}), to: {to_dev_name} (type: {to_dev_type}, is_fw: {is_to_firewall})", file=sys.stderr, flush=True)

                                        # Extract interface information from hop (for firewalls)
                                        # Interfaces should be from the firewall device's perspective
                                        in_interface = None
                                        out_interface = None

                                        if is_from_firewall or is_to_firewall:
                                            # Determine which device is the firewall
                                            firewall_dev = None
                                            if is_from_firewall:
                                                firewall_dev = from_dev
                                            elif is_to_firewall:
                                                firewall_dev = to_dev

                                            # Debug: print hop keys to see what's available
                                            print(f"DEBUG: Firewall detected! Firewall device: {firewall_dev.get('devName') if isinstance(firewall_dev, dict) else 'unknown'}", file=sys.stderr, flush=True)
                                            print(f"DEBUG: Hop keys: {list(hop.keys())}", file=sys.stderr, flush=True)
                                            print(f"DEBUG: Branch keys: {list(branch.keys())}", file=sys.stderr, flush=True)

                                            # Print full structures for debugging
                                            print(f"DEBUG: Full hop structure: {json.dumps(hop, indent=2, default=str)}", file=sys.stderr, flush=True)
                                            print(f"DEBUG: Full branch structure: {json.dumps(branch, indent=2, default=str)}", file=sys.stderr, flush=True)

                                            # Try various possible field names for interfaces
                                            # Check hop level first - these should be the firewall's interfaces
                                            in_interface = (
                                                hop.get("inInterface") or
                                                hop.get("inIntf") or
                                                hop.get("inputInterface") or
                                                hop.get("fromIntf") or
                                                hop.get("inboundInterface") or
                                                hop.get("inInterfaceName") or
                                                hop.get("inboundIntf") or
                                                hop.get("fromInterface") or
                                                hop.get("fromInterfaceName") or
                                                hop.get("in_interface") or
                                                hop.get("input_interface")
                                            )

                                            out_interface = (
                                                hop.get("outInterface") or
                                                hop.get("outIntf") or
                                                hop.get("outputInterface") or
                                                hop.get("toIntf") or
                                                hop.get("outboundInterface") or
                                                hop.get("outInterfaceName") or
                                                hop.get("outboundIntf") or
                                                hop.get("toInterface") or
                                                hop.get("toInterfaceName") or
                                                hop.get("out_interface") or
                                                hop.get("output_interface")
                                            )

                                            # Check firewall device object for interface information
                                            if isinstance(firewall_dev, dict):
                                                if not in_interface:
                                                    in_interface = (
                                                        firewall_dev.get("inInterface") or
                                                        firewall_dev.get("inIntf") or
                                                        firewall_dev.get("inputInterface") or
                                                        firewall_dev.get("interface") or
                                                        firewall_dev.get("intf") or
                                                        firewall_dev.get("interfaceName") or
                                                        firewall_dev.get("inInterfaceName")
                                                    )
                                                if not out_interface:
                                                    out_interface = (
                                                        firewall_dev.get("outInterface") or
                                                        firewall_dev.get("outIntf") or
                                                        firewall_dev.get("outputInterface") or
                                                        firewall_dev.get("interface") or
                                                        firewall_dev.get("intf") or
                                                        firewall_dev.get("interfaceName") or
                                                        firewall_dev.get("outInterfaceName")
                                                    )

                                            # Also check branch level for interface information
                                            if not in_interface:
                                                in_interface = (
                                                    branch.get("inInterface") or
                                                    branch.get("inIntf") or
                                                    branch.get("inputInterface") or
                                                    branch.get("inInterfaceName") or
                                                    branch.get("fromIntf") or
                                                    branch.get("in_interface") or
                                                    branch.get("input_interface")
                                                )
                                            if not out_interface:
                                                out_interface = (
                                                    branch.get("outInterface") or
                                                    branch.get("outIntf") or
                                                    branch.get("outputInterface") or
                                                    branch.get("outInterfaceName") or
                                                    branch.get("toIntf") or
                                                    branch.get("out_interface") or
                                                    branch.get("output_interface")
                                                )

                                            print(f"DEBUG: Extracted interfaces for firewall {firewall_dev.get('devName') if isinstance(firewall_dev, dict) else 'unknown'} - In: {in_interface}, Out: {out_interface}", file=sys.stderr, flush=True)

                                        # Only add if we have device information
                                        if from_dev_name != "Unknown" or to_dev_name:
                                            hop_info = {
                                                "hop_sequence": hop.get("sequnce", hop.get("sequence", len(simplified_hops))),
                                                "from_device": from_dev_name,
                                                "to_device": to_dev_name,
                                                "from_device_type": map_device_type(
                                                    from_dev.get("devType", "") if isinstance(from_dev, dict) else "",
                                                    device_type_map,
                                                    device_name=from_dev_name if from_dev_name != "Unknown" else None
                                                ),
                                                "to_device_type": map_device_type(
                                                    to_dev.get("devType", "") if isinstance(to_dev, dict) else "",
                                                    device_type_map,
                                                    device_name=to_dev_name if to_dev_name else None
                                                ),
                                                "status": branch_status,
                                                "failure_reason": branch_failure_reason
                                            }

                                            # Add firewall interface information if device is a firewall
                                            # Based on debug output analysis:
                                            # - When firewall is "to" device: out_interface is the firewall's IN interface
                                            # - When firewall is "from" device: in_interface is the firewall's OUT interface
                                            if is_from_firewall or is_to_firewall:
                                                firewall_device_name = from_dev_name if is_from_firewall else to_dev_name

                                                if is_to_firewall:
                                                    # Firewall is the "to" device
                                                    # The out_interface from this hop is the firewall's IN interface
                                                    if out_interface:
                                                        hop_info["in_interface"] = out_interface
                                                        print(f"DEBUG: Firewall {firewall_device_name} (as 'to') - IN interface from out_interface: {out_interface}", file=sys.stderr, flush=True)

                                                if is_from_firewall:
                                                    # Firewall is the "from" device - its egress is the interface toward "to"
                                                    # API may put firewall's out in in_interface (from-device interface) or out_interface
                                                    fw_out = in_interface or out_interface
                                                    if fw_out:
                                                        hop_info["out_interface"] = fw_out
                                                        print(f"DEBUG: Firewall {firewall_device_name} (as 'from') - OUT interface: {fw_out} (in_interface={in_interface}, out_interface={out_interface})", file=sys.stderr, flush=True)

                                                hop_info["is_firewall"] = True
                                                hop_info["firewall_device"] = firewall_device_name

                                            simplified_hops.append(hop_info)

                                    # If branch has failure reason, use it for path-level
                                    if branch_failure_reason and not path_failure_reason:
                                        path_failure_reason = branch_failure_reason

                    # Post-process: Combine firewall interfaces from multiple hops
                    # A firewall appears in two hops (as "to" and "from"), and we need to combine them
                    # Pattern from debug: When firewall is "to", out_interface is firewall's IN
                    #                    When firewall is "from", in_interface is firewall's OUT
                    def _interface_normalize(val):
                        """Get comparable string from in/out_interface (dict or str)."""
                        if val is None:
                            return ""
                        if isinstance(val, dict):
                            s = val.get("intfDisplaySchemaObj", {}).get("value") or val.get("PhysicalInftName") or val.get("name") or val.get("value") or ""
                        else:
                            s = str(val).strip()
                        s = (s or "").lower()
                        for prefix in ("ethernet", "eth"):
                            if s.startswith(prefix):
                                s = s[len(prefix):].lstrip("/")
                                break
                        return s
                    firewall_interface_map = {}  # Map firewall device name to its complete interface info
                    for hop_info in simplified_hops:
                        if hop_info.get("is_firewall"):
                            fw_name = hop_info.get("firewall_device")
                            if fw_name:
                                if fw_name not in firewall_interface_map:
                                    firewall_interface_map[fw_name] = {}
                                # Merge interface information
                                if "in_interface" in hop_info:
                                    firewall_interface_map[fw_name]["in_interface"] = hop_info["in_interface"]
                                if "out_interface" in hop_info:
                                    firewall_interface_map[fw_name]["out_interface"] = hop_info["out_interface"]
                    def _infer_egress_interface(ingress_val):
                        """Infer egress (1/2 when in is 1/1, 1/1 when in is 1/2) to match port-check display."""
                        n = _interface_normalize(ingress_val)
                        if n == "1/1":
                            return "ethernet1/2"
                        if n == "1/2":
                            return "ethernet1/1"
                        return None
                    # Don't show same interface for both in and out (API sometimes returns only one)
                    for fw_name, fw_info in firewall_interface_map.items():
                        in_val = fw_info.get("in_interface")
                        out_val = fw_info.get("out_interface")
                        if in_val is not None and out_val is not None and _interface_normalize(in_val) == _interface_normalize(out_val):
                            fw_info["out_interface"] = None
                            print(f"DEBUG: Server - Cleared duplicate out_interface for {fw_name} (same as in_interface)", file=sys.stderr, flush=True)
                        # Infer egress when missing so path query matches port-check (1/1 -> 1/2)
                        if in_val is not None and fw_info.get("out_interface") is None:
                            inferred = _infer_egress_interface(in_val)
                            if inferred:
                                fw_info["out_interface"] = inferred
                                print(f"DEBUG: Server - Inferred out_interface for {fw_name}: {inferred}", file=sys.stderr, flush=True)

                    # Update all firewall hops with complete interface information
                    for hop_info in simplified_hops:
                        if hop_info.get("is_firewall"):
                            fw_name = hop_info.get("firewall_device")
                            if fw_name and fw_name in firewall_interface_map:
                                # Use the combined interface info
                                if "in_interface" in firewall_interface_map[fw_name]:
                                    hop_info["in_interface"] = firewall_interface_map[fw_name]["in_interface"]
                                if "out_interface" in firewall_interface_map[fw_name]:
                                    hop_info["out_interface"] = firewall_interface_map[fw_name]["out_interface"]
                                print(f"DEBUG: Final interfaces for {fw_name}: In={hop_info.get('in_interface')}, Out={hop_info.get('out_interface')}", file=sys.stderr, flush=True)

                    return simplified_hops, path_status_overall, path_failure_reason
                except Exception as e:
                    print(f"DEBUG: Error extracting path hops from {source_name}: {str(e)}", file=sys.stderr, flush=True)
                    import traceback
                    print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                    return [], "Unknown", None

            # Try to extract path details from Step 3 response first
            simplified_hops = []
            path_status_overall = "Unknown"
            path_failure_reason = None

            if path_data is not None:
                # Extract simplified hop information: device name, status, and reason
                simplified_hops, path_status_overall, path_failure_reason = await extract_path_hops(path_data, "Step 3 response")

                # Query Panorama for security zones for firewall interfaces
                if simplified_hops:
                    await _add_panorama_zones_to_hops(simplified_hops)
                    await _add_panorama_device_groups_to_hops(simplified_hops)

                # If we found hops, use simplified format
                if simplified_hops:
                    # Clean hops: JSON-serializable values; in_interface/out_interface as display name only (Ethernet1/1 style)
                    def _interface_display_name(val):
                        if val is None:
                            return None
                        if isinstance(val, str):
                            return val.strip() or None
                        if isinstance(val, dict):
                            return (
                                val.get("intfDisplaySchemaObj", {}).get("value")
                                or val.get("PhysicalInftName")
                                or val.get("name")
                                or val.get("value")
                            ) or None
                        return str(val) if val else None
                    def _normalize_interface_caps(s):
                        """Display firewall interfaces as Ethernet1/1, Ethernet1/2."""
                        if not s or not isinstance(s, str):
                            return s
                        s = s.strip()
                        if not s:
                            return None
                        low = s.lower()
                        if low.startswith("ethernet"):
                            return "Ethernet" + s[8:]
                        if low.startswith("eth"):
                            return "Ethernet" + s[3:]
                        return s[0].upper() + s[1:] if len(s) > 0 else s
                    cleaned_hops = []
                    for hop in simplified_hops:
                        cleaned_hop = {}
                        for k, v in hop.items():
                            if k in ("in_interface", "out_interface"):
                                raw = _interface_display_name(v)
                                cleaned_hop[k] = _normalize_interface_caps(raw) if raw else None
                            elif v is None:
                                cleaned_hop[k] = None
                            elif isinstance(v, (str, int, float, bool)):
                                cleaned_hop[k] = v
                            else:
                                cleaned_hop[k] = str(v)
                        cleaned_hops.append(cleaned_hop)
                    result["path_hops"] = cleaned_hops
                    result["path_status"] = path_status_overall
                    # Filter out noisy NetBrain status descriptions (e.g. "L2 connections has not been discovered")
                    _raw_desc = path_data.get("statusDescription", path_failure_reason or "") or ""
                    _noise_phrases = ["l2 connections has not been discovered", "l2 connection has not been discovered"]
                    if any(p in _raw_desc.lower() for p in _noise_phrases):
                        _raw_desc = ""
                    result["path_status_description"] = _raw_desc
                    if path_failure_reason:
                        result["path_failure_reason"] = path_failure_reason or ""
                else:
                    # Fallback: Don't store full path_data (it might be too large)
                    # Set message so the chat UI shows a friendly summary instead of raw tables
                    result["message"] = f"Path from {source_trimmed} to {destination_trimmed} was calculated. Hop-by-hop details could not be extracted from the API response; you can view the full path in the NetBrain UI."
                    result["note"] = f"Path details available via taskID '{task_id}'."
                    print(f"DEBUG: Could not extract hops from Step 3, not storing full path_data to avoid large response", file=sys.stderr, flush=True)
            else:
                # If Step 3 didn't return data, try to extract from Step 2 response (calc_data)
                # Sometimes live data returns path details directly in the calculation response
                print(f"DEBUG: Step 3 returned no data, checking Step 2 response for path details", file=sys.stderr, flush=True)
                simplified_hops, path_status_overall, path_failure_reason = await extract_path_hops(calc_data, "Step 2 response")

                # Query Panorama for security zones and device groups for firewall hops
                if simplified_hops:
                    await _add_panorama_zones_to_hops(simplified_hops)
                    await _add_panorama_device_groups_to_hops(simplified_hops)

                if simplified_hops:
                    # Clean hops: in_interface/out_interface as display name only (Ethernet1/1 style)
                    def _interface_display_name(val):
                        if val is None:
                            return None
                        if isinstance(val, str):
                            return val.strip() or None
                        if isinstance(val, dict):
                            return (
                                val.get("intfDisplaySchemaObj", {}).get("value")
                                or val.get("PhysicalInftName")
                                or val.get("name")
                                or val.get("value")
                            ) or None
                        return str(val) if val else None
                    def _normalize_interface_caps(s):
                        if not s or not isinstance(s, str):
                            return s
                        s = s.strip()
                        if not s:
                            return None
                        low = s.lower()
                        if low.startswith("ethernet"):
                            return "Ethernet" + s[8:]
                        if low.startswith("eth"):
                            return "Ethernet" + s[3:]
                        return s[0].upper() + s[1:] if len(s) > 0 else s
                    cleaned_hops = []
                    for hop in simplified_hops:
                        cleaned_hop = {}
                        for k, v in hop.items():
                            if k in ("in_interface", "out_interface"):
                                raw = _interface_display_name(v)
                                cleaned_hop[k] = _normalize_interface_caps(raw) if raw else None
                            elif v is None:
                                cleaned_hop[k] = None
                            elif isinstance(v, (str, int, float, bool)):
                                cleaned_hop[k] = v
                            else:
                                cleaned_hop[k] = str(v)
                        cleaned_hops.append(cleaned_hop)
                    result["path_hops"] = cleaned_hops
                    result["path_status"] = path_status_overall
                    # Filter out noisy NetBrain status descriptions (e.g. "L2 connections has not been discovered")
                    _raw_desc2 = calc_data.get("statusDescription", path_failure_reason or "") or ""
                    _noise_phrases2 = ["l2 connections has not been discovered", "l2 connection has not been discovered"]
                    if any(p in _raw_desc2.lower() for p in _noise_phrases2):
                        _raw_desc2 = ""
                    result["path_status_description"] = _raw_desc2
                    if path_failure_reason:
                        result["path_failure_reason"] = path_failure_reason or ""
                    print(f"DEBUG: Successfully extracted {len(simplified_hops)} hops from Step 2 response", file=sys.stderr, flush=True)
                else:
                    # Path calculated but hop details could not be extracted - set message for chat UI
                    result["message"] = f"Path from {source_trimmed} to {destination_trimmed} was calculated. Hop-by-hop details could not be extracted; view the full path in the NetBrain UI."
                    result["note"] = f"Path details available via taskID '{task_id}'."
                    if step3_error:
                        result["step3_info"] = f"Path details endpoint returned: {step3_error}. This is optional - path calculation was successful."
                    # Also include calc_data for debugging
                    result["calc_data_keys"] = list(calc_data.keys()) if isinstance(calc_data, dict) else "Not a dict"

            # LLM analysis disabled for path queries to avoid MCP serialization issues
            # (This was added with other MCP tools and may be causing the crash)

            # Return the result directly - ensure None values in string fields become empty strings
            # path_hops are already cleaned (converted to basic JSON types) so include them
            final_result = {}
            try:
                for key, value in result.items():
                    if key in ["statusDescription", "path_status_description", "path_failure_reason", "gateway_used"] and value is None:
                        final_result[key] = ""
                    else:
                        final_result[key] = value

                # Include path_hops_count for convenience (even if path_hops is included)
                if "path_hops" in final_result and isinstance(final_result["path_hops"], list):
                    final_result["path_hops_count"] = len(final_result["path_hops"])

                # Test serialization before returning
                json.dumps(final_result, default=str)
                print(f"DEBUG: Final result is JSON-serializable, returning", file=sys.stderr, flush=True)
                print(f"DEBUG: Final result keys: {list(final_result.keys())}", file=sys.stderr, flush=True)
                return final_result
            except Exception as e:
                print(f"DEBUG: Error preparing result for return: {e}", file=sys.stderr, flush=True)
                import traceback
                print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                # Return minimal error result
                return {
                    "error": f"Error preparing result: {str(e)}",
                    "source": source_trimmed if 'source_trimmed' in locals() else source,
                    "destination": destination_trimmed if 'destination_trimmed' in locals() else destination,
                    "taskID": task_id if 'task_id' in locals() else None
                }

    except aiohttp.ClientError as e:
        print(f"DEBUG: aiohttp.ClientError in _query_network_path_impl: {str(e)}", file=sys.stderr, flush=True)
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": f"Network error: {str(e)}"}
    except asyncio.TimeoutError:
        print(f"DEBUG: asyncio.TimeoutError in _query_network_path_impl", file=sys.stderr, flush=True)
        return {"error": "Request timed out. Please try again later."}
    except Exception as e:
        print(f"DEBUG: Unexpected exception in _query_network_path_impl: {str(e)}", file=sys.stderr, flush=True)
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}


def _denying_firewall_from_hops(path_hops: list) -> Optional[str]:
    """From path_hops, return the hostname of the firewall that denied traffic (last Failed hop with a firewall)."""
    if not path_hops:
        return None
    for hop in reversed(path_hops):
        if hop.get("status") != "Failed":
            continue
        fr = (hop.get("failure_reason") or "").lower()
        if "policy" not in fr and "denied" not in fr and "deny" not in fr:
            continue
        from_type = (hop.get("from_device_type") or "").lower()
        to_type = (hop.get("to_device_type") or "").lower()
        if "palo alto" in from_type or "firewall" in from_type:
            return hop.get("from_device")
        if "palo alto" in to_type or "firewall" in to_type:
            return hop.get("to_device")
    # Fallback: last firewall in path (denial often at last FW)
    for hop in reversed(path_hops):
        from_type = (hop.get("from_device_type") or "").lower()
        to_type = (hop.get("to_device_type") or "").lower()
        if "palo alto" in from_type or "firewall" in from_type:
            return hop.get("from_device")
        if "palo alto" in to_type or "firewall" in to_type:
            return hop.get("to_device")
    return None


async def _check_path_allowed_impl(
    source: str,
    destination: str,
    protocol: str,
    port: str,
    is_live: int = 1
):
    """Internal implementation: run path calc with continue_on_policy_denial=False and interpret result."""
    print(f"DEBUG: _check_path_allowed_impl called with source={source}, destination={destination}, protocol={protocol}, port={port}, is_live={is_live}", file=sys.stderr, flush=True)
    try:
        path_result = await _query_network_path_impl(
            source=source,
            destination=destination,
            protocol=protocol,
            port=port,
            is_live=is_live,
            continue_on_policy_denial=False
        )
        if "error" in path_result:
            return {
                "source": source,
                "destination": destination,
                "protocol": protocol,
                "port": port,
                "status": "unknown",
                "reason": f"Unable to check path: {path_result.get('error')}",
                "path_exists": False,
                "error": path_result.get("error")
            }
        status_code = path_result.get("statusCode")
        status_description = path_result.get("statusDescription", "")
        path_status = path_result.get("path_status", "")
        path_failure_reason = path_result.get("path_failure_reason", "")
        path_hops = path_result.get("path_hops", [])
        path_hops_count = path_result.get("path_hops_count", 0)
        gateway_used = path_result.get("gateway_used")
        if status_code == 790200:
            if path_hops_count > 0 and path_hops:
                if path_status == "Success" or (not path_status or path_status == ""):
                    return {
                        "source": source,
                        "destination": destination,
                        "protocol": protocol,
                        "port": port,
                        "status": "allowed",
                        "reason": "Path exists and traffic is allowed by policy",
                        "path_exists": True,
                        "path_hops_count": path_hops_count,
                        "path_hops": path_hops,
                        "gateway_used": gateway_used,
                        "status_code": status_code,
                        "status_description": status_description
                    }
                elif path_status == "Failed":
                    firewall_denied_by = _denying_firewall_from_hops(path_hops)
                    if path_failure_reason and ("policy" in path_failure_reason.lower() or "denied" in path_failure_reason.lower() or "deny" in path_failure_reason.lower()):
                        return {
                            "source": source,
                            "destination": destination,
                            "protocol": protocol,
                            "port": port,
                            "status": "denied",
                            "reason": f"Traffic is denied by policy: {path_failure_reason}",
                            "path_exists": True,
                            "path_hops_count": path_hops_count,
                            "path_hops": path_hops,
                            "gateway_used": gateway_used,
                            "policy_details": path_failure_reason,
                            "firewall_denied_by": firewall_denied_by,
                            "status_code": status_code,
                            "status_description": status_description
                        }
                    return {
                        "source": source,
                        "destination": destination,
                        "protocol": protocol,
                        "port": port,
                        "status": "unknown",
                        "reason": path_failure_reason or "Path calculation failed but no specific reason provided",
                        "path_exists": False,
                        "path_hops_count": path_hops_count,
                        "path_hops": path_hops,
                        "gateway_used": gateway_used,
                        "status_code": status_code,
                        "status_description": status_description
                    }
            return {
                "source": source,
                "destination": destination,
                "protocol": protocol,
                "port": port,
                "status": "denied",
                "reason": "No path found - traffic may be denied by policy or path does not exist",
                "path_exists": False,
                "path_hops_count": 0,
                "path_hops": path_hops,
                "gateway_used": gateway_used,
                "status_code": status_code,
                "status_description": status_description
            }
        return {
            "source": source,
            "destination": destination,
            "protocol": protocol,
            "port": port,
            "status": "denied",
            "reason": f"Path calculation returned status code {status_code}: {status_description}. Traffic is likely denied by policy.",
            "path_exists": False,
            "path_hops": path_hops,
            "gateway_used": gateway_used,
            "status_code": status_code,
            "status_description": status_description
        }
    except Exception as e:
        print(f"DEBUG: Error in _check_path_allowed_impl: {str(e)}", file=sys.stderr, flush=True)
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {
            "source": source,
            "destination": destination,
            "protocol": protocol,
            "port": port,
            "status": "unknown",
            "reason": f"Error checking path: {str(e)}",
            "path_exists": False,
            "error": str(e)
        }


# ---------------------------------------------------------------------------
# MCP tool functions
# ---------------------------------------------------------------------------

# Register a tool with the MCP server using the @mcp.tool() decorator
@mcp.tool()
async def query_network_path(
    source: str,
    destination: str,
    protocol: str,
    port: str,
    is_live: int = 1,
    continue_on_policy_denial: bool = True
):
    """
    Get the network path (hop-by-hop) between source and destination using NetBrain.

    Use this tool when the user wants to see the path or route between two IPs (which devices/hops traffic takes).
    Typical phrases: "find path from X to Y", "show path from X to Y", "network path between X and Y", "trace path".
    Input: source (IP), destination (IP); optionally protocol and port.
    Do NOT use for: "is path allowed" or "is traffic allowed" (use check_path_allowed). Do NOT use for rack or device lookups (use get_rack_details or get_device_rack_location).

    Examples: "find path from 10.0.0.1 to 10.0.1.1", "network path between 192.168.1.1 and 192.168.2.1"

    **Query variations (all → query_network_path; need source and destination IPs; do NOT use for "is path allowed" → use check_path_allowed):**
    - "find path from 10.0.0.1 to 10.0.1.1" / "show path from 10.0.0.1 to 10.0.1.1"
    - "network path between 10.0.0.1 and 10.0.1.1" / "path from 10.0.0.1 to 10.0.1.1"
    - "trace path 10.0.0.1 to 10.0.1.1" / "get path from 10.0.0.1 to 10.0.1.1"
    - "how does traffic get from 10.0.0.1 to 10.0.1.1?" / "route from 10.0.0.1 to 10.0.1.1"
    - "show me the path/hops from 10.0.0.1 to 10.0.1.1" / "path hops from X to Y"

    **HANDLING FOLLOW-UP RESPONSES:**
    - If conversation history shows a previous clarification question was asked in the standard format: "What would you like to do with [IP]? 1) Query Panorama for object groups, 2) Look up device in NetBox, 3) Look up rack in NetBox, 4) Query network path"
    - AND the current query is just "4" or "four" → this means the user selected option 4 (Query network path)
    - **CRITICAL: The standard clarification question order is: 1) Panorama, 2) Device, 3) Rack, 4) Network Path - if you see "4" and the question lists "4) Query network path", use this tool**
    - Note: Network path queries require both source and destination IPs, so you may need to ask for the destination IP if only one IP is in the conversation history

    This function follows the three-step process from NetBrain API documentation:
    1. Resolve device gateway (GET /V1/CMDB/Path/Gateways)
    2. Calculate path (POST /V1/CMDB/Path/Calculation)
    3. Get path details (GET /V1/CMDB/Path/Calculation/{taskID}/OverView)

    Args:
        source: Source IP address or hostname (e.g., "192.168.1.1")
        destination: Destination IP address (e.g., "192.168.1.100")
        protocol: Network protocol to query (e.g., "TCP" or "UDP")
        port: Port number to query (e.g., "80", "443", "22")
        is_live: Use live data (0=Baseline, 1=Live access, default=1)
        continue_on_policy_denial: Continue calculation even if denied by device-level or subnet-level policy (default=True)

    Returns:
        dict: Network path information including:
            - source: Source endpoint
            - destination: Destination endpoint
            - protocol: Protocol used
            - port: Port number
            - taskID: Task ID from NetBrain API
            - path_details: Detailed hop-by-hop path information
            - ai_analysis: Optional AI-enhanced analysis (if LLM available)
            - error: Error message if query fails
    """
    return await _query_network_path_impl(source, destination, protocol, port, is_live, continue_on_policy_denial)


# Same pattern as query_network_path: thin MCP wrapper that delegates to internal impl
@mcp.tool()
async def check_path_allowed(
    source: str,
    destination: str,
    protocol: str,
    port: str,
    is_live: int = 1
):
    """
    check_path_allowed: Use ONLY for "is path allowed", "traffic allowed", or "check if traffic from A to B is allowed" — two IPs in the query; set source=first IP, destination=second IP; never use Panorama or rack tools for these.

    Check if traffic from source IP to destination IP is allowed or denied by policy (NetBrain). Parameters: source, destination, protocol (e.g. TCP), port (e.g. 443). Do not use for: which group contains an IP (query_panorama_ip_object_group), list IPs in a group (query_panorama_address_group_members), rack (get_rack_details), device (get_device_rack_location), path hops (query_network_path).

    Examples:
    - "Is path allowed from 10.0.0.1 to 10.0.1.1?" → source="10.0.0.1", destination="10.0.1.1"
    - "Check if traffic from 10.0.0.1 to 10.0.1.1 on TCP 443 is allowed" → source="10.0.0.1", destination="10.0.1.1", protocol="TCP", port="443"

    **Query variations (all → check_path_allowed; need TWO IPs; do NOT use for device/rack lookups):**
    - "is path allowed from 10.0.0.1 to 10.0.1.1?" / "is traffic allowed from 10.0.0.1 to 10.0.1.1?"
    - "check if traffic from X to Y is allowed" / "can traffic from X reach Y?"
    - "path allowed 10.0.0.1 to 10.0.1.1" / "traffic allowed 10.0.0.1 10.0.1.1"
    - "does path exist from 10.0.0.1 to 10.0.1.1?" / "is connectivity allowed from X to Y?"
    - "check path allowed from 10.0.0.1 to 10.0.1.1 on TCP 443"

    This function uses NetBrain Path Calculation API with policy enforcement enabled:
    - Sets continue_on_policy_denial=False to stop calculation when policy denies the path
    - Analyzes the result to determine if the path is allowed or denied
    - Returns a clear status: "allowed", "denied", or "unknown"

    Args:
        source: Source IP address or hostname (e.g., "192.168.1.1")
        destination: Destination IP address (e.g., "192.168.1.100")
        protocol: Network protocol to check (e.g., "TCP" or "UDP")
        port: Port number to check (e.g., "80", "443", "22")
        is_live: Use live data (0=Baseline, 1=Live access, default=1)

    Returns:
        dict: Policy check result including:
            - source: Source endpoint
            - destination: Destination endpoint
            - protocol: Protocol checked
            - port: Port checked
            - status: "allowed", "denied", or "unknown"
            - reason: Explanation of the status
            - path_exists: Whether a path exists (even if denied)
            - policy_details: Additional policy information if available
            - error: Error message if check fails
    """
    return await _check_path_allowed_impl(source, destination, protocol, port, is_live)
