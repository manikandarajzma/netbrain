"""
MCP Server for NetBrain Network Query
FastMCP-based server that provides network path querying capabilities.

This module:
- Exposes a query_network_path tool via MCP protocol
- Connects to NetBrain API for network path information
- Follows the three-step Path Calculation API process:
  1. Resolve device gateway
  2. Calculate path
  3. Get path details
- Optionally uses Ollama LLM for AI-enhanced analysis
- Handles authentication and error handling
"""

# Import aiohttp for asynchronous HTTP client operations
# This allows making async HTTP requests to the NetBrain API
import aiohttp

# Import asyncio for asynchronous programming support
# Used for async/await syntax and async function definitions
import asyncio

# Import json for JSON serialization/deserialization
# Used to parse API responses and format LLM prompts
import json

# Import ssl for SSL/TLS context configuration
# Used to disable SSL verification for self-signed certificates
import ssl

# Import type hints for better code documentation and IDE support
# Optional: indicates a value can be None
# Dict, Any, List: type hints for dictionaries, any type, and lists
from typing import Optional, Dict, Any, List

# Import the local netbrainauth module for authentication
# This module handles getting access tokens from NetBrain API
import netbrainauth

# Import the local panoramaauth module for Panorama API integration
# This module handles querying security zones from Panorama
import panoramaauth

# Import FastMCP for creating MCP servers easily
# FastMCP provides decorators and utilities for MCP server development
from fastmcp import FastMCP

# Import ChatOllama for LLM integration with Ollama
# This allows AI-enhanced analysis of network path data
from langchain_ollama import ChatOllama

# Import LangChain prompt templates for structured prompt management
# ChatPromptTemplate provides better maintainability, reusability, and variable substitution
from langchain_core.prompts import ChatPromptTemplate

# Import os for environment variable access
# Used to read NETBRAIN_URL from environment variables
import os
import sys

# Disable SSL warnings from urllib3
# This suppresses warnings about unverified SSL certificates
import urllib3

# Disable all SSL warnings globally
# This prevents console spam when using self-signed certificates
urllib3.disable_warnings()

# Initialize MCP server instance
# FastMCP("netbrain-mcp-server") creates a new MCP server with the given name
# This name is used for identification in MCP protocol communication
mcp = FastMCP("netbrain-mcp-server")

# Initialize LLM for AI analysis (lazy initialization)
# ChatOllama is lazy and doesn't connect until first use, so we can initialize it here
# We'll test the connection when it's first needed
mcp.llm = None  # Will be initialized lazily when needed
mcp.llm_error = None  # Store initialization error for debugging

def _get_llm():
    """Get or initialize the LLM instance (lazy initialization)."""
    if mcp.llm is None:
        try:
            print("DEBUG: Initializing LLM (ChatOllama)...", file=sys.stderr, flush=True)
            llm = ChatOllama(
                model="llama3.2:latest",  # Ollama model identifier (use 'model' not 'model_name')
                temperature=0.7,  # Sampling temperature (0.0 = deterministic, 1.0 = very random)
                base_url="http://localhost:11434",  # Explicit Ollama URL
            )
            mcp.llm = llm
            mcp.llm_error = None  # Clear any previous error
            print("DEBUG: LLM initialized successfully", file=sys.stderr, flush=True)
        except Exception as e:
            # If LLM initialization fails (e.g., Ollama not running), set to None
            # The server will still work but without AI-enhanced analysis
            error_msg = str(e)
            error_traceback = None
            try:
                import traceback
                error_traceback = traceback.format_exc()
            except:
                pass
            print(f"DEBUG: LLM initialization failed: {error_msg}", file=sys.stderr, flush=True)
            if error_traceback:
                print(f"DEBUG: LLM initialization traceback: {error_traceback}", file=sys.stderr, flush=True)
            mcp.llm = False  # Use False to indicate failed initialization (different from None = not tried)
            mcp.llm_error = {
                "error": error_msg,
                "traceback": error_traceback
            }
    return mcp.llm if mcp.llm is not False else None

# NetBrain API configuration
# Get NETBRAIN_URL from environment variable, default to localhost if not set
# os.getenv() reads environment variables, second parameter is the default value
# This should match the URL used in netbrainauth.py
NETBRAIN_URL = os.getenv("NETBRAIN_URL", "http://localhost")

# NetBox API configuration
# NETBOX_URL: Base URL for NetBox (hardcoded)
# NETBOX_TOKEN: API token for NetBox authentication (hardcoded)
# NETBOX_VERIFY_SSL: Set to "false" to disable SSL verification (default true)
NETBOX_URL = "http://192.168.15.136:8080".rstrip("/")
NETBOX_TOKEN = "f652dc1564700a3a90aabfa903a8a61db6ea007f"
NETBOX_VERIFY_SSL = os.getenv("NETBOX_VERIFY_SSL", "true").lower() in ["1", "true", "yes"]


# Cache for device type mappings (numeric code -> name)
_device_type_cache: Optional[Dict[int, str]] = None

# Cache for device name -> type name mappings (from Devices API)
_device_name_to_type_cache: Optional[Dict[str, str]] = None

# Debug info for Devices API call (to be included in result)
_devices_api_debug_info: Optional[Dict[str, Any]] = None

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


@mcp.tool()
async def get_rack_details(
    rack_name: str,
    site_name: Optional[str] = None,
    format: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Get rack details from NetBox including rack information and devices in the rack.
    
    **CRITICAL: When to use this tool:**
    - Use this tool ONLY when the query contains a RACK NAME (a SHORT identifier like "A1", "A4", "B2")
    - Rack names are SHORT (1-3 characters, typically letter + number, NO dashes)
    - Examples of rack names: "A1", "A4", "B2", "Rack A4"
    - **CRITICAL: If query mentions "space utilization", "utilization", "rack details", "rack" with a SHORT name (like "A4") → this is ALWAYS a rack query → use this tool**
    
    **Rack name identification:**
    - Rack names are SHORT (1-3 characters)
    - Rack names do NOT contain dashes (-)
    - Pattern: letter(s) + number(s), e.g., "A1", "A4", "B12"
    - If you see "A1" or "A4" → this is a RACK NAME → use this tool
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
    - Query: "A1 in Round Rock DC" → rack_name="A1", site_name="Round Rock DC", format="table"
    - Query: "show rack A4" → rack_name="A4", site_name=None, format="table"
    - Query: "A1" → rack_name="A1", site_name=None, format="table"
    - Query: "space utilization of A4" → rack_name="A4", site_name=None, format="table" (A4 is a rack name, NOT a device!)
    - Query: "rack A4 utilization" → rack_name="A4", site_name=None, format="table"
    - Query: "A4 space usage" → rack_name="A4", site_name=None, format="table"
    
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
    rack_name = (rack_name or "").strip()
    if not rack_name:
        return {"error": "Rack name cannot be empty"}
    
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
            # Try exact name match first
            params = {"name": rack_name_clean}
            async with session.get(url, headers=headers, params=params, ssl=ssl_context, timeout=15) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {
                        "error": f"NetBox rack lookup failed: HTTP {response.status}",
                        "details": error_text[:500]
                    }
                data = await response.json()
        except Exception as e:
            return {"error": f"NetBox rack lookup error: {str(e)}"}
        
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results:
            # Fallback to generic search, with site filter if provided
            try:
                params = {"q": rack_name_clean}
                if site_id:
                    params["site_id"] = site_id
                async with session.get(url, headers=headers, params=params, ssl=ssl_context, timeout=15) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "error": f"NetBox rack search failed: HTTP {response.status}",
                            "details": error_text[:500]
                        }
                    data = await response.json()
                    results = data.get("results", []) if isinstance(data, dict) else []
            except Exception as e:
                return {"error": f"NetBox rack search error: {str(e)}"}
        
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
            # Get unique sites from matching racks
            sites = {}
            for candidate in matching_racks:
                site = candidate.get("site") or {}
                site_id_val = site.get("id") if isinstance(site, dict) else site
                site_name_val = site.get("name") or site.get("display") if isinstance(site, dict) else site
                if site_id_val and site_id_val not in sites:
                    sites[site_id_val] = site_name_val
            
            # If multiple sites and no site filter provided, return ambiguity
            if len(sites) > 1 and not site_id:
                return {
                    "rack": rack_name_clean,
                    "error": f"Multiple racks named '{rack_name_clean}' found at different sites",
                    "sites": list(sites.values()),
                    "requires_site": True
                }
            
            # Use first matching rack
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
    conversation_history: Optional[List[Dict[str, str]]] = None
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
    - Query: "all racks in Round Rock DC" → site_name="Round Rock DC", format="table"
    
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
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Get device information from NetBox including rack location, device details, or specific fields.
    
    **CRITICAL: When to use this tool:**
    - Use this tool when the query contains a DEVICE NAME (a name with DASHES, e.g., "roundrock-dc-border-leaf1", "leander-dc-leaf1")
    - Device names are LONG strings with multiple dashes separating parts
    - Examples of device names: "roundrock-dc-border-leaf1", "leander-dc-border-leaf2", "roundrock-dc-leaf1"
    
    **Device name identification:**
    - Device names ALWAYS contain DASHES (-)
    - Device names are typically 15+ characters long
    - If you see "roundrock-dc-border-leaf1" → this is a DEVICE NAME → use this tool
    - If you see "A1" or "A4" → this is a RACK NAME → use get_rack_details instead
    
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

    Args:
        device_name: The FULL device name to look up (e.g., "roundrock-dc-border-leaf1" - must include all parts with dashes)
        intent: What information to return - "device_details" (all info), "rack_location_only", "device_type_only", "status_only", "site_only", "manufacturer_only"
        format: Output format - "table" (recommended), "json", "list", or None for natural language summary
        conversation_history: Optional conversation history for context-aware responses

    Returns:
        dict: Device information based on intent - all details, or specific field(s) as requested
    """
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
        
        return cleaned_result


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
                
                print(f"DEBUG: Server - Collected interfaces for {fw_name}: {firewall_interface_map[fw_name]['interfaces']}", file=sys.stderr, flush=True)
                
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
                print(f"DEBUG: Server - Adding zones to {len(fw_data['hops'])} hops for {fw_name}", file=sys.stderr, flush=True)
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
                    
                    print(f"DEBUG: Server - Matching zones for {fw_name}: in_intf_name={in_intf_name}, out_intf_name={out_intf_name}, zones={zones}", file=sys.stderr, flush=True)
                    
                    # Match zones with case-insensitive interface name matching
                    if in_intf_name:
                        # Try exact match first
                        if in_intf_name in zones and zones[in_intf_name]:
                            hop_info["in_zone"] = zones[in_intf_name]
                            print(f"DEBUG: Server - Set in_zone for {fw_name} hop to {zones[in_intf_name]} (exact match)", file=sys.stderr, flush=True)
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
                                print(f"DEBUG: Server - Set in_zone for {fw_name} hop to {matched_zone} (case-insensitive match)", file=sys.stderr, flush=True)
                    
                    if out_intf_name:
                        # Try exact match first
                        if out_intf_name in zones and zones[out_intf_name]:
                            hop_info["out_zone"] = zones[out_intf_name]
                            print(f"DEBUG: Server - Set out_zone for {fw_name} hop to {zones[out_intf_name]} (exact match)", file=sys.stderr, flush=True)
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
                                print(f"DEBUG: Server - Set out_zone for {fw_name} hop to {matched_zone} (case-insensitive match)", file=sys.stderr, flush=True)
                
                print(f"DEBUG: Zones for {fw_name}: {zones}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"DEBUG: Error querying Panorama for {fw_name}: {str(e)}", file=sys.stderr, flush=True)
                import traceback
                print(f"DEBUG: Panorama query traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)


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
        print(f"DEBUG: Server - No firewalls found in hops for device group query", file=sys.stderr, flush=True)
        return
    
    # Query Panorama for device groups
    try:
        firewall_list = list(firewall_names)
        print(f"DEBUG: Server - Querying device groups for firewalls: {firewall_list}", file=sys.stderr, flush=True)
        
        device_groups = await panoramaauth.get_device_groups_for_firewalls(
            firewall_names=firewall_list
        )
        
        print(f"DEBUG: Server - Device groups returned: {device_groups}", file=sys.stderr, flush=True)
        
        # Add device group information to firewall hops
        for fw_name, hops in firewall_hop_map.items():
            device_group = device_groups.get(fw_name)
            if device_group:
                print(f"DEBUG: Server - Adding device group '{device_group}' to {len(hops)} hops for {fw_name}", file=sys.stderr, flush=True)
                for hop_info in hops:
                    hop_info["device_group"] = device_group
                    print(f"DEBUG: Server - Set device_group for {fw_name} hop to {device_group}", file=sys.stderr, flush=True)
            else:
                print(f"DEBUG: Server - No device group found for {fw_name}", file=sys.stderr, flush=True)
    
    except Exception as e:
        print(f"DEBUG: Error querying Panorama device groups: {str(e)}", file=sys.stderr, flush=True)
        import traceback
        print(f"DEBUG: Panorama device group query traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)


# Register a tool with the MCP server using the @mcp.tool() decorator
# This makes the function callable via MCP protocol from clients
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
    Query network path between source and destination using NetBrain Path Calculation API.
    
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
    # Declare globals at the top of the function
    global _device_name_to_type_cache, _devices_api_debug_info
    
    # Debug: Print function entry and parameters
    print(f"DEBUG: query_network_path called with source={source}, destination={destination}, protocol={protocol}, port={port}, is_live={is_live}, continue_on_policy_denial={continue_on_policy_denial}", file=sys.stderr, flush=True)
    
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
    
    # Prepare HTTP headers for all API requests
    # NetBrain API uses "Token" header (capital T) for authentication
    headers = {
        "Content-Type": "application/json",  # Indicates we're sending JSON data
        "Accept": "application/json",  # Indicates we want JSON response
        "Token": auth_token  # NetBrain API uses "Token" header for authentication
    }
    
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
            
            async with session.get(gateway_url, headers=headers, params=gateway_params, ssl=ssl_context) as gateway_response:
                response_status = gateway_response.status
                print(f"DEBUG: Step 1 - Response status: {response_status}", file=sys.stderr, flush=True)
                
                source_gateway = None  # Initialize to None
                
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
            print(f"DEBUG: Step 3 - Attempting to get path details for taskID: {task_id} (optional, file=sys.stderr, flush=True)")
            
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
                                print(f"DEBUG: Step 3 - Path details retrieved on attempt {attempt} (status {path_response.status}, file=sys.stderr, flush=True)")
                                
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
            result = {
                "source": source_trimmed,
                "destination": destination_trimmed,
                "protocol": protocol,
                "port": port,
                "taskID": task_id,
                "statusCode": calc_data.get("statusCode"),
                "statusDescription": calc_data.get("statusDescription"),
                "gateway_used": source_gateway.get("gatewayName"),
                "path_info": calc_data  # Original calculation response
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
                                        
                                        from_dev_name = from_dev.get("devName", "Unknown")
                                        to_dev_name = to_dev.get("devName") if to_dev.get("devName") else None
                                        
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
                                                    # Firewall is the "from" device
                                                    # The in_interface from this hop is the firewall's OUT interface
                                                    if in_interface:
                                                        hop_info["out_interface"] = in_interface
                                                        print(f"DEBUG: Firewall {firewall_device_name} (as 'from') - OUT interface from in_interface: {in_interface}", file=sys.stderr, flush=True)
                                                
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
                    result["path_hops"] = simplified_hops
                    result["path_status"] = path_status_overall
                    result["path_status_description"] = path_data.get("statusDescription", path_failure_reason or "")
                    if path_failure_reason:
                        result["path_failure_reason"] = path_failure_reason
                    # Debug: Add cache status to result for troubleshooting
                    result["_debug_device_cache_size"] = len(_device_name_to_type_cache) if _device_name_to_type_cache else 0
                    result["_debug_device_cache_sample"] = list(_device_name_to_type_cache.keys())[:5] if _device_name_to_type_cache else []
                    # Add Devices API debug info if available
                    if _devices_api_debug_info:
                        result["_debug_devices_api"] = _devices_api_debug_info
                else:
                    # Fallback to full path_data if we couldn't parse it
                    result["path_details"] = path_data
                    print(f"DEBUG: Could not extract hops from Step 3, storing full path_data", file=sys.stderr, flush=True)
            else:
                # If Step 3 didn't return data, try to extract from Step 2 response (calc_data)
                # Sometimes live data returns path details directly in the calculation response
                print(f"DEBUG: Step 3 returned no data, checking Step 2 response for path details", file=sys.stderr, flush=True)
                simplified_hops, path_status_overall, path_failure_reason = await extract_path_hops(calc_data, "Step 2 response")
                
                # Query Panorama for security zones for firewall interfaces
                if simplified_hops:
                    await _add_panorama_zones_to_hops(simplified_hops)
                    await _add_panorama_device_groups_to_hops(simplified_hops)
                
                if simplified_hops:
                    result["path_hops"] = simplified_hops
                    result["path_status"] = path_status_overall
                    result["path_status_description"] = calc_data.get("statusDescription", path_failure_reason or "")
                    if path_failure_reason:
                        result["path_failure_reason"] = path_failure_reason
                    # Debug: Add cache status to result for troubleshooting
                    result["_debug_device_cache_size"] = len(_device_name_to_type_cache) if _device_name_to_type_cache else 0
                    result["_debug_device_cache_sample"] = list(_device_name_to_type_cache.keys())[:5] if _device_name_to_type_cache else []
                    # Add Devices API debug info if available
                    if _devices_api_debug_info:
                        result["_debug_devices_api"] = _devices_api_debug_info
                    print(f"DEBUG: Successfully extracted {len(simplified_hops)} hops from Step 2 response", file=sys.stderr, flush=True)
                else:
                    # Add note about using taskID to query path details separately
                    result["note"] = f"Path calculation succeeded. Use taskID '{task_id}' to query detailed path information separately if needed."
                    if step3_error:
                        result["step3_info"] = f"Path details endpoint returned: {step3_error}. This is optional - path calculation was successful."
                    # Also include calc_data for debugging
                    result["calc_data_keys"] = list(calc_data.keys()) if isinstance(calc_data, dict) else "Not a dict"
            
            # Try to enhance with LLM analysis if available (lazy initialization)
            llm = _get_llm()
            if llm is not None:
                try:
                    # Use LangChain ChatPromptTemplate for structured prompt management
                    # This provides better maintainability, reusability, and variable substitution
                    analysis_prompt_template = ChatPromptTemplate.from_messages([
                        ("system", """You are a network analysis assistant. Analyze the network path information and provide:
                        1. A summary of the path
                        2. Connectivity status
                        3. Key devices in the path
                        4. Any potential issues or recommendations
                        
                        Format your response as a JSON object with these fields:
                        {{
                            "summary": "string",
                            "connectivity": "string",
                            "key_devices": ["string"],
                            "recommendations": ["string"]
                        }}"""),
                        ("human", "Analyze this network path:\n{path_data}")
                    ])
                    
                    # Format the prompt with the path data
                    formatted_messages = analysis_prompt_template.format_messages(
                        path_data=json.dumps(result, indent=2)
                    )
                    
                    # Invoke the LLM with the formatted prompt
                    llm_response = await llm.ainvoke(formatted_messages)
                    
                    # Extract content from the response
                    if hasattr(llm_response, 'content'):
                        response_content = llm_response.content
                    else:
                        response_content = str(llm_response)
                    
                    # Try to parse as JSON, fallback to string if not valid JSON
                    try:
                        analysis = json.loads(response_content)
                    except json.JSONDecodeError:
                        # If response is not valid JSON, wrap it in a summary field
                        analysis = {"summary": response_content}
                    
                    result["ai_analysis"] = analysis
                except Exception as e:
                    # Log the error for debugging but don't fail the entire request
                    print(f"DEBUG: LLM analysis failed: {str(e)}", file=sys.stderr, flush=True)
                    import traceback
                    print(f"DEBUG: LLM analysis traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                    pass
            
            return result
            
    except aiohttp.ClientError as e:
        return {"error": f"Network error: {str(e)}"}
    except asyncio.TimeoutError:
        return {"error": "Request timed out. Please try again later."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


# Check if this script is being run directly (not imported as a module)
# __name__ will be "__main__" when the script is executed directly
# This allows the script to be both runnable and importable
if __name__ == "__main__":
    # Redirect stderr to a log file for easier debugging
    # Logs will be written to mcp_server.log in the same directory as this script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, "mcp_server.log")
    log_file = open(log_file_path, "a", encoding="utf-8")
    # Create a wrapper that writes to both stderr and the log file
    class TeeStderr:
        def __init__(self, file, stderr):
            self.file = file
            self.stderr = stderr
        def write(self, text):
            self.stderr.write(text)
            self.file.write(text)
            self.file.flush()
        def flush(self):
            self.stderr.flush()
            self.file.flush()
    sys.stderr = TeeStderr(log_file, sys.__stderr__)
    print(f"DEBUG: Server logs will be written to: {log_file_path}", file=sys.__stderr__, flush=True)
    
    # Run the MCP server using stdio transport
    # stdio transport means the server communicates via standard input/output
    # This is the standard way MCP servers communicate with clients
    mcp.run(transport="stdio")
