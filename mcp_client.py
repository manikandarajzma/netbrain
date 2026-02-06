"""
MCP Client for NetBrain Network Query
Streamlit-based web interface for querying network paths via MCP server.

This module provides a user-friendly web interface where users can:
- Enter source and destination IP addresses or hostnames
- Select protocol (TCP or UDP)
- Enter port number
- Query network paths and display results
"""

# Import streamlit library for creating web UI components (forms, buttons, displays)
import streamlit as st

# Import asyncio for handling asynchronous operations (needed for MCP client)
import asyncio
import sys
from typing import Optional
import warnings

# Suppress asyncio cleanup warnings on Windows (these are harmless cleanup errors)
if sys.platform == 'win32':
    # Suppress ConnectionResetError in asyncio cleanup callbacks on Windows
    def _suppress_asyncio_cleanup_warnings():
        import logging
        logging.getLogger('asyncio').setLevel(logging.ERROR)
    
    # Set up warning filter for asyncio cleanup errors
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='asyncio')

# Import ClientSession for managing MCP client connections (for stdio fallback)
from mcp import ClientSession
# Try to import FastMCP Client for HTTP transport (preferred)
try:
    from fastmcp import Client as FastMCPClient
    FASTMCP_CLIENT_AVAILABLE = True
except ImportError:
    FASTMCP_CLIENT_AVAILABLE = False
    print("DEBUG: FastMCP Client not available, will use stdio transport", file=sys.stderr, flush=True)

# Fallback to stdio if HTTP client not available
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
try:
    from mcp.shared.exceptions import McpError
except ImportError:
    # Fallback if McpError is not available
    McpError = Exception

# Import ChatOllama for LLM integration (currently imported but not actively used in client)
from langchain_ollama import ChatOllama
# Import Pydantic for structured outputs
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("DEBUG: Pydantic not available, falling back to manual JSON parsing", file=sys.stderr, flush=True)

# Import pandas for reading spreadsheet files (CSV, Excel)
import pandas as pd

# Import json for serialization
import json

# Import requests for fetching images
try:
    import requests
    from io import BytesIO
    from PIL import Image
    IMAGE_FETCH_AVAILABLE = True
except ImportError:
    IMAGE_FETCH_AVAILABLE = False
    print("DEBUG: PIL/requests not available, elevation images will show as links only", file=sys.stderr, flush=True)

# Import matplotlib and networkx for graph visualization
import matplotlib.pyplot as plt
import networkx as nx

# Configure Streamlit page settings:
# - page_title: Sets the browser tab title to "NetBrain Network Query"
# - page_icon: Sets the browser tab icon to a globe emoji (🌐)
# - layout: Sets the page layout to "centered" for better visual presentation
st.set_page_config(
    page_title="NetBrain Network Query",
    page_icon="🌐",
    layout="centered"
)

def extract_interface_name(interface_data):
    """
    Extract interface name from interface data structure.
    Handles both string values and dictionary structures from NetBrain API.
    
    Args:
        interface_data: Can be a string (interface name) or dict with interface info
        
    Returns:
        str: Interface name (e.g., "ethernet1/1") or None
    """
    if not interface_data:
        return None
    
    import re
    
    # If it's already a string, check if it's a string representation of a dict
    if isinstance(interface_data, str):
        interface_str = interface_data.strip()
        
        # If it's a plain interface name (doesn't look like a dict), return it
        if not interface_str.startswith('{') and not interface_str.startswith("'"):
            # Check if it contains dict-like patterns
            if "'intfKeyObj'" not in interface_str and '"intfKeyObj"' not in interface_str:
                if "'intfDisplaySchemaObj'" not in interface_str and '"intfDisplaySchemaObj"' not in interface_str:
                    # Looks like a plain interface name
                    return interface_str
        
        # It's a dict string representation - try to extract interface name
        # Priority 1: Look for PhysicalInftName (most reliable)
        patterns = [
            r"['\"]PhysicalInftName['\"]\s*:\s*['\"]([^'\"]+)['\"]",  # PhysicalInftName
            r"PhysicalInftName['\"]?\s*:\s*['\"]?([a-zA-Z0-9/_-]+)",  # More flexible
        ]
        for pattern in patterns:
            match = re.search(pattern, interface_str)
            if match:
                val = match.group(1).strip()
                if val and val not in ['', 'None', 'null']:
                    return val
        
        # Priority 2: Look for 'value': 'ethernetX/Y' (prefer ethernet interfaces)
        ethernet_patterns = [
            r"['\"]value['\"]\s*:\s*['\"](ethernet[^'\"]+)['\"]",  # ethernet interfaces
            r"value['\"]?\s*:\s*['\"]?(ethernet[a-zA-Z0-9/_-]+)",  # More flexible
        ]
        for pattern in ethernet_patterns:
            match = re.search(pattern, interface_str)
            if match:
                val = match.group(1).strip()
                if val and val not in ['', 'None', 'null']:
                    return val
        
        # Priority 3: Look for any 'value': 'something' 
        value_patterns = [
            r"['\"]value['\"]\s*:\s*['\"]([^'\"]+)['\"]",  # Standard value
            r"value['\"]?\s*:\s*['\"]?([a-zA-Z0-9/_-]+)",  # More flexible
        ]
        for pattern in value_patterns:
            match = re.search(pattern, interface_str)
            if match:
                val = match.group(1).strip()
                # Skip empty values, None, null, or schema strings
                if val and val not in ['', 'None', 'null', 'schema']:
                    return val
        
        # Priority 4: Try to parse as JSON
        try:
            import json
            # Try with single quotes replaced (Python repr format)
            json_str = interface_str.replace("'", '"')
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                interface_data = parsed  # Fall through to dict handling
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, try ast.literal_eval for Python repr format
            try:
                import ast
                parsed = ast.literal_eval(interface_str)
                if isinstance(parsed, dict):
                    interface_data = parsed  # Fall through to dict handling
            except (ValueError, SyntaxError):
                # Can't parse - return None to indicate failure
                return None
    
    # If it's a dictionary, try to extract the interface name
    if isinstance(interface_data, dict):
        # Try PhysicalInftName first (as seen in the UI)
        if 'PhysicalInftName' in interface_data:
            val = interface_data['PhysicalInftName']
            if val and str(val).strip() and str(val).strip() not in ['None', 'null', '']:
                return str(val).strip()
        
        # Try intfKeyObj.value (NetBrain out_interface structure)
        if 'intfKeyObj' in interface_data and isinstance(interface_data['intfKeyObj'], dict):
            val = interface_data['intfKeyObj'].get('value')
            if val and str(val).strip() and str(val).strip() not in ['None', 'null', '', 'schema']:
                return str(val).strip()
        
        # Try intfDisplaySchemaObj.value
        if 'intfDisplaySchemaObj' in interface_data and isinstance(interface_data['intfDisplaySchemaObj'], dict):
            val = interface_data['intfDisplaySchemaObj'].get('value')
            if val and str(val).strip() and str(val).strip() not in ['None', 'null', '', 'schema']:
                return str(val).strip()
        
        # Try common field names
        for field in ['name', 'interface', 'intf', 'interfaceName', 'intfName', 'value']:
            if field in interface_data:
                val = interface_data[field]
                if val and str(val).strip() and str(val).strip() not in ['None', 'null', '']:
                    return str(val).strip()
    
    # If we can't extract it, return None
    return None


def _normalize_interface_for_compare(name):
    """
    Normalize interface name for comparison so 'ethernet1/1' and '1/1' are treated as the same.
    """
    if not name or not isinstance(name, str):
        return ""
    s = name.strip().lower()
    for prefix in ("ethernet", "eth"):
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip("/")
            break
    return s or name.strip().lower()


def infer_egress_interface(ingress_name):
    """
    Infer the common egress interface when API only returns ingress (e.g. 1/1 -> 1/2).
    Matches port-check behavior: 1/1 is typically inside, 1/2 outside.
    Returns the pair interface (ethernet1/1 <-> ethernet1/2) or None if not inferrable.
    """
    if not ingress_name or not isinstance(ingress_name, str):
        return None
    n = _normalize_interface_for_compare(ingress_name)
    if n == "1/1":
        return ingress_name.replace("1/2", "1/1").replace("1/1", "1/2")  # 1/1 -> 1/2, preserve ethernet
    if n == "1/2":
        return ingress_name.replace("1/1", "1/2").replace("1/2", "1/1")  # 1/2 -> 1/1
    return None


def get_device_icon_path(device_type: str) -> Optional[str]:
    """
    Get icon image file path for device type based on NetBrain UI conventions.
    
    Args:
        device_type: Device type name (e.g., "Palo Alto Firewall", "Arista Switch")
    
    Returns:
        str: Path to icon image file, or None if not found
    """
    if not device_type:
        return None
    
    import os
    device_type_lower = device_type.lower()
    
    # Base directory for icons (create if needed)
    icons_dir = os.path.join(os.path.dirname(__file__), "icons")
    
    # Map device types to icon filenames
    icon_mapping = {
        "palo alto": "paloalto_firewall.png",
        "paloalto": "paloalto_firewall.png",
        "pan-": "paloalto_firewall.png",
        "arista": "arista_switch.png",
        "cisco": "cisco_device.png",
        "juniper": "juniper_device.png",
        "switch": "generic_switch.png",
        "router": "generic_router.png",
        "firewall": "generic_firewall.png",
    }
    
    # Find matching icon
    for key, filename in icon_mapping.items():
        if key in device_type_lower:
            icon_path = os.path.join(icons_dir, filename)
            if os.path.exists(icon_path):
                return icon_path
    
    # Default icon
    default_path = os.path.join(icons_dir, "default_device.png")
    if os.path.exists(default_path):
        return default_path
    
    return None

def create_device_icon_image(device_type: str, size: int = 64) -> Optional[Image.Image]:
    """
    Create a programmatic icon image for device type.
    Falls back to this if icon files don't exist.
    For devices without specific types, creates a simple colored square.
    
    Args:
        device_type: Device type name
        size: Icon size in pixels
    
    Returns:
        PIL Image or None
    """
    if not IMAGE_FETCH_AVAILABLE:
        return None
    
    try:
        from PIL import Image, ImageDraw
        
        # Create a colored square icon based on device type
        device_type_lower = device_type.lower() if device_type else ""
        
        # Determine color based on device type
        if "palo alto" in device_type_lower or "paloalto" in device_type_lower or "pan-" in device_type_lower or "firewall" in device_type_lower:
            # Red for firewalls (Palo Alto style)
            bg_color = (220, 50, 50)  # Red
        elif "arista" in device_type_lower:
            # Blue for Arista switches
            bg_color = (33, 150, 243)  # Blue
        elif "cisco" in device_type_lower:
            # Light blue for Cisco
            bg_color = (100, 181, 246)  # Light blue
        elif "switch" in device_type_lower:
            # Blue for switches
            bg_color = (33, 150, 243)  # Blue
        elif "router" in device_type_lower:
            # Green for routers
            bg_color = (76, 175, 80)  # Green
        else:
            # Gray for default/unknown devices - simple square, no text
            bg_color = (158, 158, 158)  # Gray
        
        # Create image with transparency support - simple square, no text
        img = Image.new('RGBA', (size, size), (*bg_color, 255))
        
        # For unknown devices, just return the simple square
        # For known device types without icon files, also return simple square
        # (This keeps it clean - only actual icon files will have detailed graphics)
        if not device_type_lower or device_type_lower == "":
            # Unknown device - simple gray square
            return img
        
        # For known device types, return colored square (they should have icon files)
        # If we're here, it means no icon file was found, so use simple colored square
        return img
    except Exception as e:
        print(f"DEBUG: Error creating device icon: {e}", file=sys.stderr, flush=True)
        return None

def create_path_graph(path_hops, source, destination):
    """
    Create a network graph visualization of the path hops using matplotlib and networkx.
    
    Args:
        path_hops: List of hop dictionaries with from_device, to_device, status, failure_reason
        source: Source IP/device name
        destination: Destination IP/device name
    
    Returns:
        matplotlib figure object or None if no valid hops
    """
    if not path_hops or len(path_hops) == 0:
        return None
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Track all unique devices
    devices = set()
    edges = []
    
    # Track device types for each device
    device_types = {}  # {device_name: 'device_type'}
    
    # Track device IP addresses
    device_ips = {}  # {device_name: 'ip_address'}
    
    # Try to infer device type for source/destination if they're IPs
    # (They won't be in path_hops, so we need to handle them separately)
    def infer_device_type_from_name(device_name):
        """Infer device type from device name patterns"""
        if not device_name:
            return None
        name_lower = device_name.lower()
        if 'fw' in name_lower or 'firewall' in name_lower or 'palo' in name_lower:
            return 'Palo Alto Firewall'
        elif 'switch' in name_lower or 'sw' in name_lower:
            if 'arista' in name_lower:
                return 'Arista Switch'
            return 'Switch'
        elif 'router' in name_lower or 'rtr' in name_lower:
            return 'Router'
        # If it's an IP address, we can't infer type
        if device_name.replace('.', '').replace(':', '').replace('/', '').isdigit():
            return None
        return None
    
    # Track device groups for all devices (not just firewalls)
    device_groups = {}  # {device_name: 'device_group'}
    
    # Track firewall interfaces, zones, and device groups for overlay
    firewall_interfaces = {}  # {device_name: {'in': 'interface', 'out': 'interface', 'in_zone': 'zone', 'out_zone': 'zone', 'device_group': 'group'}}
    
    # Add source node
    if source:
        devices.add(source)
        # Try to infer device type for source
        inferred_type = infer_device_type_from_name(source)
        if inferred_type:
            device_types[source] = inferred_type
        print(f"DEBUG: Added source node: {source} (type: {inferred_type})", file=sys.stderr, flush=True)
    
    # Process each hop - filter out invalid hops (None/null values)
    print(f"DEBUG: create_path_graph - Processing {len(path_hops)} hops", file=sys.stderr, flush=True)
    hops_processed = 0
    hops_skipped = 0
    
    for hop in path_hops:
        from_dev = hop.get('from_device', 'Unknown')
        to_dev = hop.get('to_device')
        status = hop.get('status', 'Unknown')
        failure_reason = hop.get('failure_reason')
        
        print(f"DEBUG: Hop - from: '{from_dev}', to: '{to_dev}', status: '{status}', failure_reason: '{failure_reason}'", file=sys.stderr, flush=True)
        
        # Skip hops with invalid/None/null device names
        # Filter out None, 'None', empty strings, and 'Unknown' as from_device (unless it's the actual source)
        if not from_dev or from_dev in [None, 'None', 'null', ''] or (from_dev == 'Unknown' and from_dev != source):
            print(f"DEBUG: Skipping hop - invalid from_device: '{from_dev}'", file=sys.stderr, flush=True)
            hops_skipped += 1
            continue
        
        # Handle None to_dev - use destination if available, otherwise skip
        if not to_dev or to_dev in [None, 'None', 'null', '']:
            if destination:
                # Use destination as the to_device for the last hop
                to_dev = destination
                print(f"DEBUG: Using destination '{destination}' as to_device for hop", file=sys.stderr, flush=True)
            else:
                print(f"DEBUG: Skipping hop - invalid to_device: '{to_dev}' and no destination", file=sys.stderr, flush=True)
                hops_skipped += 1
                continue
        
        # Don't skip failed hops - show them with different styling
        # This allows users to see what NetBrain discovered, even if incomplete
        is_failed = (status == 'Failed' or failure_reason)
        if is_failed:
            print(f"DEBUG: Hop is failed, but will display with warning style", file=sys.stderr, flush=True)
        
        hops_processed += 1
        
        # Collect device type information
        from_device_type = hop.get('from_device_type', '')
        to_device_type = hop.get('to_device_type', '')
        
        # Debug: Print device types being collected
        if from_device_type or to_device_type:
            print(f"DEBUG: Graph - Device types from hop: from='{from_dev}' -> type='{from_device_type}', to='{to_dev}' -> type='{to_device_type}'", file=sys.stderr, flush=True)
        
        if from_dev and from_device_type:
            device_types[from_dev] = from_device_type
            print(f"DEBUG: Graph - Set device type for '{from_dev}': '{from_device_type}'", file=sys.stderr, flush=True)
        if to_dev and to_dev not in [None, 'None', 'null', ''] and to_device_type:
            device_types[to_dev] = to_device_type
            print(f"DEBUG: Graph - Set device type for '{to_dev}': '{to_device_type}'", file=sys.stderr, flush=True)
        
        # Extract IP addresses if available
        from_dev_ip = hop.get('from_device_ip') or hop.get('fromDev', {}).get('devIP') if isinstance(hop.get('fromDev'), dict) else None
        to_dev_ip = hop.get('to_device_ip') or hop.get('toDev', {}).get('devIP') if isinstance(hop.get('toDev'), dict) else None
        
        if from_dev and from_dev_ip:
            device_ips[from_dev] = from_dev_ip
        if to_dev and to_dev_ip:
            device_ips[to_dev] = to_dev_ip
        
        # Check if devices are firewalls and collect interface information
        is_firewall = hop.get('is_firewall', False)
        if is_firewall:
            firewall_device = hop.get('firewall_device')
            if not firewall_device:
                # Determine firewall device name
                if 'fw' in from_dev.lower() or 'palo' in from_dev.lower() or 'fortinet' in from_dev.lower():
                    firewall_device = from_dev
                elif to_dev and ('fw' in to_dev.lower() or 'palo' in to_dev.lower() or 'fortinet' in to_dev.lower()):
                    firewall_device = to_dev
            
            if firewall_device:
                # Extract interface names
                in_interface = hop.get('in_interface')
                out_interface = hop.get('out_interface')
                
                # Extract zone and device group information
                in_zone = hop.get('in_zone')
                out_zone = hop.get('out_zone')
                device_group = hop.get('device_group')
                
                # Debug: Print zone and device group information from hop
                print(f"DEBUG: Graph - Extracting info for {firewall_device}: in_zone={in_zone}, out_zone={out_zone}, device_group={device_group}", file=sys.stderr, flush=True)
                
                # Determine if firewall is "from" or "to" device in this hop
                from_dev = hop.get('from_device', '')
                to_dev = hop.get('to_device', '')
                is_firewall_from = (from_dev == firewall_device)
                is_firewall_to = (to_dev == firewall_device)
                
                # Use extract_interface_name helper to get clean interface names
                in_intf_name = extract_interface_name(in_interface) if in_interface else None
                out_intf_name = extract_interface_name(out_interface) if out_interface else None
                
                # Server logic: When firewall is "from", in_interface is actually the firewall's OUT interface
                # When firewall is "to", out_interface is actually the firewall's IN interface
                # Match port-check: when we only have one interface, infer egress (1/1 <-> 1/2)
                if is_firewall_from and in_intf_name and not out_intf_name:
                    # Firewall is "from" - infer egress so we don't show same as ingress
                    out_intf_name = infer_egress_interface(in_intf_name) or in_intf_name
                    if out_intf_name != in_intf_name:
                        print(f"DEBUG: Graph - Firewall {firewall_device} is 'from' device, inferred out_interface: {out_intf_name}", file=sys.stderr, flush=True)
                    else:
                        print(f"DEBUG: Graph - Firewall {firewall_device} is 'from' device, using in_interface as out_interface: {out_intf_name}", file=sys.stderr, flush=True)
                elif is_firewall_from and in_intf_name and out_intf_name and _normalize_interface_for_compare(in_intf_name) == _normalize_interface_for_compare(out_intf_name):
                    # Same in/out from API - infer distinct egress (match port-check display)
                    out_intf_name = infer_egress_interface(in_intf_name) or out_intf_name
                    print(f"DEBUG: Graph - Firewall {firewall_device} inferred distinct out_interface: {out_intf_name}", file=sys.stderr, flush=True)
                elif is_firewall_to and out_intf_name and not in_intf_name:
                    # Firewall is "to" device - out_interface is actually the IN interface
                    in_intf_name = out_intf_name
                    out_intf_name = None  # Clear out since we used it for in
                    print(f"DEBUG: Graph - Firewall {firewall_device} is 'to' device, using out_interface as in_interface: {in_intf_name}", file=sys.stderr, flush=True)
                
                # Store interface, zone, and device group information for this firewall
                if firewall_device not in firewall_interfaces:
                    firewall_interfaces[firewall_device] = {'in': None, 'out': None, 'in_zone': None, 'out_zone': None, 'device_group': None}
                
                # Update interfaces if we have new information
                if in_intf_name:
                    # Only update if we don't have a value yet
                    if not firewall_interfaces[firewall_device]['in']:
                        firewall_interfaces[firewall_device]['in'] = in_intf_name
                        print(f"DEBUG: Graph - Set in interface for {firewall_device} to {in_intf_name}", file=sys.stderr, flush=True)
                
                if out_intf_name:
                    # Only update if we don't have a value yet
                    if not firewall_interfaces[firewall_device]['out']:
                        firewall_interfaces[firewall_device]['out'] = out_intf_name
                        print(f"DEBUG: Graph - Set out interface for {firewall_device} to {out_intf_name}", file=sys.stderr, flush=True)
                elif not out_intf_name and out_interface:
                    # Debug: log when out_interface exists but extraction failed
                    print(f"DEBUG: Graph - Failed to extract out_interface for {firewall_device}. Raw value: {out_interface}", file=sys.stderr, flush=True)
                
                # Update zones if we have new information (use 'or' to allow overwriting None)
                if in_zone:
                    firewall_interfaces[firewall_device]['in_zone'] = in_zone
                    print(f"DEBUG: Graph - Set in_zone for {firewall_device} to {in_zone}", file=sys.stderr, flush=True)
                if out_zone:
                    firewall_interfaces[firewall_device]['out_zone'] = out_zone
                    print(f"DEBUG: Graph - Set out_zone for {firewall_device} to {out_zone}", file=sys.stderr, flush=True)
                if device_group:
                    firewall_interfaces[firewall_device]['device_group'] = device_group
                    device_groups[firewall_device] = device_group  # Track for all devices
                    print(f"DEBUG: Graph - Set device_group for {firewall_device} to {device_group}", file=sys.stderr, flush=True)
        
        # Infer out_zone when we have out interface (e.g. inferred) but no out_zone - match port-check (inside <-> outside)
        for _fw, ifaces in firewall_interfaces.items():
            if ifaces.get('out') and not ifaces.get('out_zone') and ifaces.get('in_zone'):
                in_zl = (ifaces['in_zone'] or "").strip().lower()
                if in_zl == "inside":
                    ifaces['out_zone'] = "outside"
                elif in_zl == "outside":
                    ifaces['out_zone'] = "inside"
        
        # Add valid devices
        if from_dev and from_dev != 'Unknown':
            devices.add(from_dev)
        if to_dev and to_dev not in [None, 'None', 'null', '']:
            devices.add(to_dev)
        
        # Only add successful edges (green lines only for successful paths)
        if status == 'Success' and not failure_reason:
            edge_color = 'green'
            edge_style = 'solid'
            edge_width = 2.0
            
            # Add edge with attributes - only for successful hops
            if from_dev and to_dev and to_dev not in [None, 'None', 'null', '']:
                edges.append((from_dev, to_dev, {
                    'color': edge_color,
                    'style': edge_style,
                    'width': edge_width,
                    'status': status,
                    'failure_reason': failure_reason
                }))
            print(f"DEBUG: Added edge: {from_dev} -> {to_dev} (status: {status})", file=sys.stderr, flush=True)
        elif from_dev and (not to_dev or to_dev in [None, 'None', 'null', '']):
            # Last hop - connect to destination if available (use gray for incomplete/failed hops)
            if destination:
                edge_color = 'gray'
                edge_style = 'dashed'
                edge_width = 1.0
                edges.append((from_dev, destination, {
                    'color': edge_color,
                    'style': edge_style,
                    'width': edge_width,
                    'status': status,
                    'failure_reason': failure_reason
                }))
                devices.add(destination)
    
    # Add destination if not already added
    if destination and destination not in devices:
        devices.add(destination)
        # Try to infer device type for destination
        inferred_type = infer_device_type_from_name(destination)
        if inferred_type:
            device_types[destination] = inferred_type
        print(f"DEBUG: Added destination node: {destination} (type: {inferred_type})", file=sys.stderr, flush=True)
    
    # If no edges were created from hops, but we have source and destination, add a direct edge
    # This is a fallback to show at least a connection attempt
    if len(edges) == 0 and source and destination:
        print(f"DEBUG: No edges from hops, adding direct edge from source to destination as fallback", file=sys.stderr, flush=True)
        edges.append((source, destination, {
            'color': 'orange',
            'style': 'dotted',
            'width': 1.0
        }))
    
    # Add nodes and edges to graph
    print(f"DEBUG: Adding {len(devices)} nodes and {len(edges)} edges to graph", file=sys.stderr, flush=True)
    print(f"DEBUG: Devices: {list(devices)}", file=sys.stderr, flush=True)
    print(f"DEBUG: Edges: {edges}", file=sys.stderr, flush=True)
    print(f"DEBUG: Hops processed: {hops_processed}, skipped: {hops_skipped}", file=sys.stderr, flush=True)
    
    G.add_nodes_from(devices)
    for edge_data in edges:
        if len(edge_data) == 3:
            from_dev, to_dev, attrs = edge_data
            G.add_edge(from_dev, to_dev, **attrs)
        elif len(edge_data) == 2:
            from_dev, to_dev = edge_data
            G.add_edge(from_dev, to_dev)
    
    if len(G.nodes()) == 0:
        print(f"DEBUG: Graph has no nodes, returning None", file=sys.stderr, flush=True)
        return None
    
    print(f"DEBUG: Graph created with {len(G.nodes())} nodes and {len(G.edges())} edges", file=sys.stderr, flush=True)
    
    # Create figure with more vertical space for path visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Add padding at top to avoid overlap with title
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95)
    
    # Clear any existing graph state and create layout based on actual path order
    pos = {}
    
    # Build ordered device list from path_hops (following the actual path sequence)
    ordered_devices = []
    seen_devices = set()
    
    # Add source first if it exists
    if source and source in G.nodes():
        ordered_devices.append(source)
        seen_devices.add(source)
    
    # Follow the path sequence from hops
    for hop in path_hops:
        from_dev = hop.get('from_device', 'Unknown')
        to_dev = hop.get('to_device')
        
        # Skip invalid devices
        if not from_dev or from_dev in [None, 'None', 'null', ''] or from_dev == 'Unknown':
            continue
        if not to_dev or to_dev in [None, 'None', 'null', '']:
            continue
        
        # Add from_device if not seen
        if from_dev not in seen_devices and from_dev in G.nodes():
            ordered_devices.append(from_dev)
            seen_devices.add(from_dev)
        
        # Add to_device if not seen
        if to_dev not in seen_devices and to_dev in G.nodes():
            ordered_devices.append(to_dev)
            seen_devices.add(to_dev)
    
    # Add destination if not already in the list
    if destination and destination in G.nodes() and destination not in seen_devices:
        ordered_devices.append(destination)
        seen_devices.add(destination)
    
    # Add any remaining devices that weren't in the path
    for node in G.nodes():
        if node not in seen_devices:
            ordered_devices.append(node)
    
    print(f"DEBUG: Ordered devices: {ordered_devices}", file=sys.stderr, flush=True)
    
    # Position devices in a linear path from left to right
    # Use hierarchical positioning: switches on top, firewalls on bottom, others in middle
    x_pos = 0
    spacing = 3.0  # Horizontal spacing between devices
    
    y_top = 1.0      # Switches
    y_middle = 0.0  # Other devices
    y_bottom = -1.0 # Firewalls
    y_source = 1.5  # Source endpoint
    y_dest = -1.5   # Destination endpoint
    
    for device in ordered_devices:
        device_type = device_types.get(device, '').lower()
        
        # Determine Y position based on device type
        if device == source:
            y = y_source
        elif device == destination:
            y = y_dest
        elif 'switch' in device_type or 'sw' in device_type:
            y = y_top
        elif 'firewall' in device_type or 'fw' in device_type or 'palo' in device_type:
            y = y_bottom
        else:
            y = y_middle
        
        pos[device] = (x_pos, y)
        x_pos += spacing
    
    # Center the layout horizontally
    if pos:
        min_x = min([p[0] for p in pos.values()])
        max_x = max([p[0] for p in pos.values()])
        center_x = (min_x + max_x) / 2
        # Shift all positions to center
        for node in pos:
            pos[node] = (pos[node][0] - center_x, pos[node][1])
    
    # Draw nodes with icons based on device type
    node_colors = []
    node_icon_images = {}  # Store icon images for each node
    
    for node in G.nodes():
        # Determine node color
        if node == source:
            node_colors.append('#4CAF50')  # Green for source
        elif node == destination:
            node_colors.append('#FF9800')  # Orange for destination
        else:
            node_colors.append('#2196F3')  # Blue for intermediate devices
        
        # Get icon image based on device type
        device_type = device_types.get(node, '')
        icon_path = get_device_icon_path(device_type)
        
        icon_image = None
        if icon_path and IMAGE_FETCH_AVAILABLE:
            # Try to load icon from file
            try:
                from PIL import Image
                icon_image = Image.open(icon_path)
                # Convert to RGBA if needed
                if icon_image.mode != 'RGBA':
                    icon_image = icon_image.convert('RGBA')
                icon_image = icon_image.resize((64, 64), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"DEBUG: Could not load icon from {icon_path}: {e}", file=sys.stderr, flush=True)
                icon_image = None
        
        # If no icon file, create programmatic icon
        if icon_image is None and IMAGE_FETCH_AVAILABLE:
            icon_image = create_device_icon_image(device_type, size=64)  # Smaller size for cleaner look
        
        if icon_image:
            node_icon_images[node] = icon_image
    
    # Draw nodes - reduced size for cleaner look
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, 
                           alpha=0.7, ax=ax, edgecolors='black', linewidths=1.5)
    
    # Draw edges FIRST so icons appear on top
    # Only draw edges that are part of the successful path (green lines only)
    # Build edges from the ordered path sequence
    successful_edges = []
    
    # Create edges following the ordered device sequence
    for i in range(len(ordered_devices) - 1):
        from_dev = ordered_devices[i]
        to_dev = ordered_devices[i + 1]
        if from_dev in pos and to_dev in pos:
            successful_edges.append((from_dev, to_dev))
    
    # Also check graph edges that are marked as successful
    for from_dev, to_dev, data in G.edges(data=True):
        status = data.get('status', 'Unknown')
        failure_reason = data.get('failure_reason')
        
        # Only include successful edges (no red lines)
        if status == 'Success' and not failure_reason:
            if (from_dev, to_dev) not in successful_edges and from_dev in pos and to_dev in pos:
                successful_edges.append((from_dev, to_dev))
    
    # Draw only successful edges in green
    for from_dev, to_dev in successful_edges:
        if from_dev in pos and to_dev in pos:
            # Calculate curve direction based on vertical position
            from_y = pos[from_dev][1]
            to_y = pos[to_dev][1]
            
            # Use more pronounced curves for vertical connections
            if abs(from_y - to_y) > 0.5:
                # Vertical connection - use U-shaped curve
                rad = 0.3 if from_y > to_y else -0.3
            else:
                # Horizontal connection - use gentle curve
                rad = 0.1
            
            nx.draw_networkx_edges(G, pos, edgelist=[(from_dev, to_dev)], 
                                  edge_color='green', style='solid',
                                  width=2.0, alpha=0.7, arrows=True,
                                  arrowsize=15, ax=ax, connectionstyle=f'arc3,rad={rad}')
    
    # Add icon images to nodes using matplotlib's OffsetImage (AFTER edges so they're on top)
    if IMAGE_FETCH_AVAILABLE:
        try:
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            import numpy as np
            
            for node in G.nodes():
                if node in pos and node in node_icon_images:
                    x, y = pos[node]
                    icon_image = node_icon_images[node]
                    
                    # Convert PIL image to numpy array for matplotlib
                    # Handle RGBA images properly
                    if icon_image.mode == 'RGBA':
                        icon_array = np.array(icon_image)
                    else:
                        icon_array = np.array(icon_image.convert('RGBA'))
                    
                    # Create OffsetImage with smaller zoom for cleaner, less intrusive icons
                    # Node size is 2500, icon is 64x64, so zoom of 0.6-0.8 should work well
                    # Icons should be visible but not dominate the graph
                    imagebox = OffsetImage(icon_array, zoom=0.7)
                    ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0, zorder=10)
                    ax.add_artist(ab)
                    print(f"DEBUG: Added icon image for node '{node}' (type: {device_types.get(node, 'unknown')}) at position ({x:.3f}, {y:.3f})", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"DEBUG: Error adding icon images to graph: {e}", file=sys.stderr, flush=True)
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
    
    
    # Draw labels with device name, IP, and device type
    # For firewalls, position labels differently to avoid overlap with interface/zone labels
    print(f"DEBUG: Graph - Device types collected: {device_types}", file=sys.stderr, flush=True)
    
    for node in G.nodes():
        if node in pos:
            x, y = pos[node]
            node_label = node[:20] + '...' if len(node) > 20 else node
            
            # Check if this is a firewall - use different positioning
            is_firewall = node in firewall_interfaces
            device_type = device_types.get(node, '').lower()
            is_firewall_type = 'firewall' in device_type or 'fw' in device_type or 'palo' in device_type
            
            # Build label text with IP address if available
            label_parts = [node_label]
            
            # Add IP address if available
            if node in device_ips and device_ips[node]:
                label_parts.append(device_ips[node])
            
            # Add device type if available
            if node in device_types and device_types[node]:
                device_type_full = device_types[node]
                label_parts.append(device_type_full)
            
            label_text = '\n'.join(label_parts)
            
            # For firewalls, position label at the very top (above device group badge)
            # For other devices, use standard positioning
            if is_firewall or is_firewall_type:
                # Firewall labels go at the very top, above device group badge
                label_y = y + 0.50  # Above device group badge (which is at y + 0.35)
            else:
                # Standard positioning for non-firewall devices
                label_y = y + 0.25
            
            # Draw label with high z-order and better visibility
            # For firewalls, ensure hostname is at the very top above all other elements
            ax.text(x, label_y, label_text, 
                   fontsize=9, fontweight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, 
                           edgecolor='black', linewidth=1.5),
                   zorder=15)  # Highest z-order to ensure it's always visible on top
    
    # Overlay firewall interface and zone information - make them clearer and more readable
    for firewall_device, interfaces in firewall_interfaces.items():
        if firewall_device in pos:
            x, y = pos[firewall_device]
            
            # Debug: Print what we have for this firewall
            print(f"DEBUG: Graph overlay for {firewall_device}: interfaces={interfaces}", file=sys.stderr, flush=True)
            
            # Calculate vertical positions for labels - reorganize to prevent overlap
            # Hostname is at y + 0.50 (drawn above this section)
            # Device group badge below hostname
            device_group_badge_y = y + 0.35
            device_group_label_y = y + 0.25
            # In interface label below device group
            in_interface_y = y + 0.15
            # Out interface label below device
            out_interface_y = y - 0.20
            
            # Add device group badge at the very top (above everything)
            device_group = interfaces.get('device_group')
            if device_group:
                # Use different colors for different device groups
                group_colors = {
                    'roundrock': 'orange',
                    'leander': 'green',
                }
                badge_color = group_colors.get(device_group.lower(), 'orange')
                # Use first letter of device group for badge
                badge_text = device_group[0].upper() if device_group else '?'
                
                # Circular badge at top
                ax.text(x, device_group_badge_y, badge_text, 
                       fontsize=12, ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle='circle,pad=0.4', facecolor=badge_color, alpha=0.95, 
                               edgecolor='black', linewidth=2),
                       color='white', zorder=13)
                
                # Device group text label below badge
                ax.text(x, device_group_label_y, f"DG: {device_group}", 
                       fontsize=8, ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,
                               edgecolor='black', linewidth=1),
                       zorder=12)
            
            # Build In interface label with zone - positioned below device group
            if interfaces.get('in'):
                in_intf = interfaces.get('in', '')
                in_zone = interfaces.get('in_zone', '')
                
                # Build label with interface and zone on separate lines for clarity
                if in_zone:
                    in_label = f"In: {in_intf}\nZone: {in_zone}"
                else:
                    in_label = f"In: {in_intf}"
                
                # Add In interface label - larger font, better padding, clear zone display
                ax.text(x, in_interface_y, in_label, 
                       fontsize=9, ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.95, 
                               edgecolor='darkblue', linewidth=1.5),
                       zorder=11)
            
            # Build Out interface label with zone - positioned below device
            out_intf = interfaces.get('out', '')
            out_zone = interfaces.get('out_zone', '')
            print(f"DEBUG: {firewall_device} - Out interface: '{out_intf}', zone: '{out_zone}'", file=sys.stderr, flush=True)
            
            # Always show Out label, even if interface is empty (show zone if available)
            if out_intf:
                # Build label with interface and zone on separate lines for clarity
                if out_zone:
                    out_label = f"Out: {out_intf}\nZone: {out_zone}"
                else:
                    out_label = f"Out: {out_intf}"
            elif out_zone:
                # If no interface but we have zone, show zone only
                out_label = f"Out: (unknown)\nZone: {out_zone}"
            else:
                # No interface or zone - skip displaying
                out_label = None
            
            # Add Out interface label if we have something to show
            if out_label:
                ax.text(x, out_interface_y, out_label, 
                       fontsize=9, ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.95,
                               edgecolor='darkgreen', linewidth=1.5),
                       zorder=11)
    
    # Add device group badges for all devices (not just firewalls)
    for device_name, device_group in device_groups.items():
        if device_name in pos and device_name not in firewall_interfaces:
            x, y = pos[device_name]
            # Create circular badge above device
            badge_y = y + 0.15
            group_colors = {
                'roundrock': 'orange',
                'leander': 'green',
            }
            badge_color = group_colors.get(device_group.lower(), 'orange')
            badge_letter = device_group[0].upper() if device_group else '?'
            ax.text(x, badge_y, badge_letter, 
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor=badge_color, alpha=0.9, edgecolor='black', linewidth=1),
                   weight='bold', color='white', zorder=12)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Source'),
        Patch(facecolor='#2196F3', label='Intermediate Device'),
        Patch(facecolor='#FF9800', label='Destination'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Success')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    ax.set_title("Network Path Visualization", fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def display_result(result):
    """
    Display network path query result in the Streamlit UI.
    
    This function handles both error and success cases, displaying:
    - Error messages and debug information for failed queries
    - Path hops visualization for successful queries
    - Full JSON details in expandable sections
    
    Args:
        result: Dictionary containing the query result from the MCP server
    """
    if isinstance(result, dict):
        # Check if the result contains an error key
        if 'error' in result:
            # Display error message in red using st.error()
            st.error(f"Error: {result['error']}")
            # Display detailed error information if available
            if 'details' in result:
                details = result['details']
                # Try to extract statusDescription for better user experience
                if isinstance(details, str):
                    if 'statusDescription:' in details:
                        try:
                            # Extract the status description
                            desc_start = details.find('statusDescription:') + len('statusDescription:')
                            desc_text = details[desc_start:].strip()
                            # Remove any trailing statusCode or other info
                            if ',' in desc_text:
                                desc_text = desc_text.split(',')[0].strip()
                            if desc_text and desc_text != 'No description':
                                st.warning(f"ℹ️ {desc_text}")
                        except:
                            pass
                
                # Show full details in expander
                with st.expander("Error Details"):
                    if isinstance(details, str):
                        st.text(details)
                    else:
                        st.json(details)
            if 'full_response' in result:
                with st.expander("Full API Response"):
                    st.text(result['full_response'])
            if 'error_message' in result:
                st.error(f"Error Message: {result['error_message']}")
            # Show token prominently at the top if available
            if 'debug_info' in result and 'auth_token' in result.get('debug_info', {}):
                st.text(f"🔑 Token: {result['debug_info']['auth_token']}")
            if 'debug_info' in result:
                with st.expander("Debug Information"):
                    st.json(result['debug_info'])
            if 'troubleshooting' in result:
                st.info(f"💡 {result['troubleshooting']}")
            if 'payload_sent' in result:
                with st.expander("View Payload Sent"):
                    st.json(result['payload_sent'])
        else:
            # Check if path calculation was successful or failed
            path_status = result.get('path_status', 'Unknown')
            path_status_description = result.get('path_status_description', '')
            path_failure_reason = result.get('path_failure_reason', '')
            
            # Determine if path failed
            path_failed = (
                path_status == 'Failed' or 
                'Failed' in str(path_status) or 
                'failed' in str(path_status_description).lower() or
                path_failure_reason or
                (result.get('statusCode') and result.get('statusCode') != 790200)
            )
            
            if path_failed:
                # Display failure information prominently
                st.error("❌ Network Path Calculation Failed")
                
                # Display path status description if available
                if path_status_description:
                    st.error(f"**Path Status:** {path_status_description}")
                elif path_status and path_status != 'Unknown':
                    st.error(f"**Path Status:** {path_status}")
                
                # Display failure reason prominently if available
                if path_failure_reason:
                    st.error(f"**Failure Reason:** {path_failure_reason}")
                
                # Show a warning box with the failure details
                failure_details = []
                if path_status_description:
                    failure_details.append(f"Status: {path_status_description}")
                if path_failure_reason:
                    failure_details.append(f"Reason: {path_failure_reason}")
                
                if failure_details:
                    st.warning("⚠️ " + " | ".join(failure_details))
            else:
                # Display success message in green using st.success()
                st.success("✅ Query completed successfully")
            
            # Display path hops if available (simplified visual representation)
            if 'path_hops' in result:
                st.subheader("Network Path")
                
                # Display path status (if not already shown above)
                if not path_failed:
                    if path_status_description:
                        st.success(f"Path Status: {path_status_description}")
                    elif path_status and path_status != 'Unknown':
                        st.success(f"Path Status: {path_status}")
                
                # Create and display graph visualization
                try:
                    graph_fig = create_path_graph(result['path_hops'], result.get('source', 'Source'), result.get('destination', 'Destination'))
                    if graph_fig:
                        st.markdown("### Network Path Graph")
                        st.pyplot(graph_fig)
                        plt.close(graph_fig)
                except Exception as e:
                    st.warning(f"Could not generate graph visualization: {str(e)}")
                
                # Display hops visually
                st.markdown("### Path Hops")
                
                # Helper function to get device icon (matching NetBrain UI style)
                def get_device_icon(device_name):
                    """Return an icon based on device name or type, matching NetBrain UI"""
                    if not device_name or device_name == "Unknown":
                        return "🌐"  # Unknown device
                    # Check if it's an IP address (endpoint) - use network device icon
                    # IP addresses are numeric with dots/colons
                    if device_name.replace('.', '').replace(':', '').replace('/', '').isdigit():
                        return "📱"  # Network device/endpoint icon (like NetBrain's IP device)
                    # Check for router indicators - use router icon
                    if any(keyword in device_name.lower() for keyword in ['router', 'rtr', 'rt', 'gw', 'gateway']):
                        return "🖥️"  # Router/server icon (like NetBrain's router icon)
                    # Default to network device icon for other network devices
                    return "📱"  # Network device icon
                
                for i, hop in enumerate(result['path_hops']):
                    # Create a visual representation of each hop
                    col1, col2, col3 = st.columns([2, 1, 3])
                    
                    with col1:
                        # From device with icon
                        from_dev = hop.get('from_device', 'Unknown')
                        from_icon = get_device_icon(from_dev)
                        st.markdown(f"{from_icon} **{from_dev}**")
                    
                    with col2:
                        # Arrow
                        st.markdown("→")
                    
                    with col3:
                        # To device and status
                        to_dev = hop.get('to_device')
                        if to_dev:
                            to_icon = get_device_icon(to_dev)
                            st.markdown(f"{to_icon} **{to_dev}**")
                        else:
                            st.markdown("🎯 *Destination*")
                        
                        # Status and failure reason
                        status = hop.get('status', 'Unknown')
                        failure_reason = hop.get('failure_reason')
                        
                        if status == 'Failed' or failure_reason:
                            st.error(f"❌ {status}")
                            if failure_reason:
                                st.caption(f"Reason: {failure_reason}")
                        elif status == 'Success':
                            st.success(f"✓ {status}")
                        else:
                            st.info(f"Status: {status}")
                    
                    # Add separator between hops
                    if i < len(result['path_hops']) - 1:
                        st.divider()
                
                # Show full details in expander
                with st.expander("View Full Path Details"):
                    # Debug: Show device cache info if available (at the top)
                    if "_debug_device_cache_size" in result:
                        cache_size = result.get('_debug_device_cache_size', 0)
                        cache_sample = result.get('_debug_device_cache_sample', [])
                        if cache_size > 0:
                            st.success(f"✅ Device Cache: {cache_size} entries. Sample devices: {', '.join(cache_sample)}")
                        else:
                            st.warning(f"⚠️ Device Cache: {cache_size} entries (cache is empty - device types will show as numbers)")
                    st.json(result)
            else:
                # No path hops available - show summary information
                if path_failed:
                    # Already displayed failure info above, but show additional details if available
                    if 'taskID' in result:
                        st.info(f"Task ID: {result['taskID']}")
                    if 'gateway_used' in result:
                        st.info(f"Gateway Used: {result['gateway_used']}")
                else:
                    # Show basic success information
                    if 'taskID' in result:
                        st.success(f"Task ID: {result['taskID']}")
                    if 'gateway_used' in result:
                        st.success(f"Gateway: {result['gateway_used']}")
                    if path_status_description and path_status_description != 'Success.':
                        st.info(f"Status: {path_status_description}")
                
                # Always show full details in expander
                with st.expander("View Full Response Details"):
                    st.json(result)
    elif isinstance(result, str):
        # Display success message
        st.success("Query completed")
        # Display the string result as plain text
        st.text(result)
    else:
        # Display result as JSON for any other type
        st.json(result)

def get_server_url():
    """
    Get the HTTP URL for the MCP server.
    
    Returns:
        str: Server URL for HTTP transport (streamable-http uses /mcp endpoint)
    """
    return "http://127.0.0.1:8765/mcp"

async def get_mcp_session():
    """
    Get an MCP session using HTTP transport (preferred) or stdio (fallback).
    This is a generator that yields the session within the proper context.
    
    Yields:
        FastMCPClient or ClientSession: MCP client session
    """
    if FASTMCP_CLIENT_AVAILABLE:
        # Use FastMCP Client for HTTP transport (automatically handles streamable-http)
        try:
            server_url = get_server_url()
            print(f"DEBUG: Connecting to MCP server via HTTP at {server_url}...", file=sys.stderr, flush=True)
            # FastMCP Client automatically infers streamable-http from URL
            # Try to configure with longer timeout for long-running requests
            try:
                # Check if FastMCPClient accepts timeout parameters
                client = FastMCPClient(server_url, timeout=600)  # 10 minute timeout
            except TypeError:
                # If timeout parameter not supported, use default
                client = FastMCPClient(server_url)
            async with client:
                print(f"DEBUG: FastMCP HTTP client connected successfully", file=sys.stderr, flush=True)
                yield client
        except Exception as e:
            # HTTP connection failed - this is expected if server is running in stdio mode
            # Silently fall back to stdio (don't print verbose traceback as it's expected)
            print(f"DEBUG: HTTP connection unavailable (server may be in stdio mode), using stdio transport...", file=sys.stderr, flush=True)
            # Fall through to stdio fallback
    
    # Fallback to stdio transport
    server_params = get_server_params()
    print(f"DEBUG: Connecting to MCP server via stdio...", file=sys.stderr, flush=True)
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print(f"DEBUG: Stdio session initialized successfully", file=sys.stderr, flush=True)
            yield session

def get_server_params():
    """
    Create server parameters for stdio communication (fallback).
    
    This function creates the configuration needed to spawn the MCP server
    as a subprocess via stdio transport.
    
    Returns:
        StdioServerParameters: Server parameters for stdio communication
    """
    import os
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(script_dir, "mcp_server.py")
    
    # Create server parameters for stdio communication:
    # - command: The command to run (python interpreter)
    # - args: Arguments to pass (the mcp_server.py script with full path)
    # This configures the client to spawn mcp_server.py as a subprocess
    from mcp import StdioServerParameters
    return StdioServerParameters(
        command="python",  # Use Python interpreter to run the server
        args=[server_path]  # Pass mcp_server.py with full path as argument
    )

def parse_query(query_text, default_live_data=True):
    """
    Parse natural language query to extract network path parameters.
    
    Args:
        query_text: Natural language query string
        default_live_data: Default value for live data
        
    Returns:
        dict: Parsed parameters (source, destination, protocol, port, is_live)
    """
    import re
    
    # Convert query to lowercase for easier parsing
    query_lower = query_text.lower() if query_text else ""
    
    # Start with default live data setting
    is_live = default_live_data
    
    # Check for live data keywords in query text
    if any(keyword in query_lower for keyword in ['live data', 'use live', 'with live', 'live access']):
        is_live = True
    # Check for keywords that disable live data
    elif any(keyword in query_lower for keyword in ['baseline', 'no live', 'disable live', 'use baseline']):
        is_live = False
    
    # Extract IP addresses using regex
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ip_addresses = re.findall(ip_pattern, query_text) if query_text else []
    
    # Extract port number (optional - defaults to 0 if not specified)
    port_patterns = [
        r'port\s+(\d{1,5})',  # "port 80" or "port 443"
        r':(\d{1,5})(?:\s|$)',  # ":80" or ":443" (not part of IP)
    ]
    
    port = "0"
    for pattern in port_patterns:
        port_match = re.search(pattern, query_text) if query_text else None
        if port_match:
            extracted_port = port_match.group(1)
            if extracted_port and 0 <= int(extracted_port) <= 65535:
                port = extracted_port
                break
    
    # Extract protocol (TCP or UDP)
    protocol = "TCP"  # Default
    if 'udp' in query_lower:
        protocol = "UDP"
    elif 'tcp' in query_lower:
        protocol = "TCP"
    
    # Extract source and destination IPs
    source = None
    destination = None
    
    if len(ip_addresses) >= 2:
        from_pos = query_lower.find('from')
        to_pos = query_lower.find('to')
        source_pos = query_lower.find('source')
        dest_pos = query_lower.find('destination')
        
        if from_pos != -1 and to_pos != -1:
            from_ip_pos = query_text.lower().find(ip_addresses[0])
            to_ip_pos = query_text.lower().find(ip_addresses[1])
            if from_pos < from_ip_pos < to_pos < to_ip_pos:
                source = ip_addresses[0]
                destination = ip_addresses[1]
            else:
                source = ip_addresses[1]
                destination = ip_addresses[0]
        elif source_pos != -1 or dest_pos != -1:
            source = ip_addresses[0]
            destination = ip_addresses[1]
        else:
            source = ip_addresses[0]
            destination = ip_addresses[1]
    elif len(ip_addresses) == 1:
        source = ip_addresses[0]
    
    return {
        'source': source,
        'destination': destination,
        'protocol': protocol,
        'port': port,
        'is_live': is_live
    }


def parse_rack_location_query(query_text):
    """
    Parse natural language query to extract device name for rack location lookup.

    Args:
        query_text: Natural language query string

    Returns:
        dict or None: {"device_name": "<name>", "format": "<format>"} if detected, else None
    """
    import re

    if not query_text:
        return None

    query_lower = query_text.lower()
    
    # Detect format request
    format_type = None
    if "table" in query_lower or "in a table" in query_lower:
        format_type = "table"
    elif "json" in query_lower:
        format_type = "json"
    elif "list" in query_lower:
        format_type = "list"

    # Avoid ambiguity with path queries
    path_keywords = ["path", "from", "to", "source", "destination"]
    has_path_keywords = any(keyword in query_lower for keyword in path_keywords)
    
    # Device detail query patterns (more flexible)
    device_detail_patterns = [
        r"details of\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"detail of\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"info about\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"information about\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"info on\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"show\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"tell me about\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"what is\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"what's\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
    ]
    
    # Check for device detail queries first (before path check)
    for pattern in device_detail_patterns:
        match = re.search(pattern, query_text, re.IGNORECASE)
        if match:
            device_name = match.group("device").strip().strip("?.!,")
            # Remove formatting requests like "in a table format", "in table", etc.
            device_name = re.sub(r"\s+in\s+(a\s+)?(table|json|list|format).*$", "", device_name, flags=re.IGNORECASE).strip()
            if device_name and not has_path_keywords:
                result = {"device_name": device_name}
                if format_type:
                    result["format"] = format_type
                return result
    
    # If path keywords present, only proceed with explicit rack/location intent
    if has_path_keywords:
        # Allow rack intent to override path keywords
        if "rack" not in query_lower and "rack location" not in query_lower:
            return None

    # Accept rack intent or "where is <device>" phrasing
    has_rack_intent = "rack" in query_lower or "rack location" in query_lower
    has_where_is_intent = "where is" in query_lower or "where's" in query_lower
    has_location_intent = "located" in query_lower or "location" in query_lower
    if not (has_rack_intent or has_where_is_intent or has_location_intent):
        return None

    patterns = [
        r"rack location of\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"rack location for\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"rack location\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"where is\s+(?P<device>.+?)\s+located",
        r"where is\s+(?P<device>.+?)\s+in\s+rack",
        r"where is\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"where's\s+(?P<device>.+?)(?:\s+in\s+.*)?$",
        r"rack\s+position\s+of\s+(?P<device>.+?)(?:\s+in\s+.*)?$"
    ]

    for pattern in patterns:
        match = re.search(pattern, query_text, re.IGNORECASE)
        if match:
            device_name = match.group("device").strip().strip("?.!,")
            # Remove formatting requests like "in a table format", "in table", etc.
            device_name = re.sub(r"\s+in\s+(a\s+)?(table|json|list|format).*$", "", device_name, flags=re.IGNORECASE).strip()
            if device_name:
                result = {"device_name": device_name}
                if format_type:
                    result["format"] = format_type
                return result

    # Fallback: look for quoted device name
    quoted = re.search(r"['\"]([^'\"]+)['\"]", query_text)
    if quoted:
        device_name = quoted.group(1).strip()
        if device_name:
            return {"device_name": device_name}

    # Fallback: "device <name>" with rack intent
    dev_match = re.search(r"device\s+([A-Za-z0-9._-]+)", query_text, re.IGNORECASE)
    if dev_match:
        return {"device_name": dev_match.group(1)}

    return None

def extract_hops_from_path_details(path_details):
    """
    Extract path hops from path_details structure (fallback extraction in client).
    
    Args:
        path_details: Dictionary containing path_overview structure
        
    Returns:
        List of hop dictionaries or None if extraction fails
    """
    try:
        if not isinstance(path_details, dict):
            return None
        
        simplified_hops = []
        
        # Check for path_overview structure
        path_overview = path_details.get('path_overview', [])
        if not path_overview:
            return None
        
        if not isinstance(path_overview, list):
            path_overview = [path_overview]
        
        # Process each path group
        for path_group in path_overview:
            if not isinstance(path_group, dict):
                continue
                
            path_list = path_group.get('path_list', [])
            if not isinstance(path_list, list):
                path_list = [path_list] if path_list else []
                
            for path in path_list:
                if not isinstance(path, dict):
                    continue
                    
                branch_list = path.get('branch_list', [])
                if not isinstance(branch_list, list):
                    branch_list = [branch_list] if branch_list else []
                    
                for branch in branch_list:
                    if not isinstance(branch, dict):
                        continue
                        
                    hop_detail_list = branch.get('hop_detail_list', [])
                    if not isinstance(hop_detail_list, list):
                        hop_detail_list = [hop_detail_list] if hop_detail_list else []
                    
                    branch_status = branch.get('status', 'Unknown')
                    branch_failure_reason = branch.get('failureReason') or branch.get('failure_reason')
                    
                    for hop in hop_detail_list:
                        if not isinstance(hop, dict):
                            continue
                            
                        from_dev = hop.get('fromDev', {})
                        to_dev = hop.get('toDev', {})
                        
                        if not isinstance(from_dev, dict):
                            from_dev = {}
                        if not isinstance(to_dev, dict):
                            to_dev = {}
                        
                        from_dev_name = from_dev.get('devName', 'Unknown')
                        to_dev_name = to_dev.get('devName') if to_dev.get('devName') else None
                        
                        # Check if device is a firewall
                        from_dev_type = str(from_dev.get('devType', '')).lower() if isinstance(from_dev, dict) else ''
                        to_dev_type = str(to_dev.get('devType', '')).lower() if isinstance(to_dev, dict) else ''
                        
                        is_from_firewall = (
                            'firewall' in from_dev_type or 
                            'fw' in from_dev_type or
                            'fw' in from_dev_name.lower() or  # Check device name for "fw"
                            'palo' in from_dev_name.lower() or
                            'fortinet' in from_dev_name.lower() or
                            'checkpoint' in from_dev_name.lower() or
                            'asa' in from_dev_name.lower()
                        )
                        
                        is_to_firewall = (
                            to_dev_name and (
                                'firewall' in to_dev_type or 
                                'fw' in to_dev_type or
                                'fw' in to_dev_name.lower() or  # Check device name for "fw"
                                'palo' in to_dev_name.lower() or
                                'fortinet' in to_dev_name.lower() or
                                'checkpoint' in to_dev_name.lower() or
                                'asa' in to_dev_name.lower()
                            )
                        )
                        
                        # Extract interface information (for firewalls)
                        in_interface = None
                        out_interface = None
                        
                        if is_from_firewall or is_to_firewall:
                            in_interface = (
                                hop.get('inInterface') or 
                                hop.get('inIntf') or 
                                hop.get('inputInterface') or
                                hop.get('fromIntf') or
                                hop.get('inboundInterface') or
                                (from_dev.get('interface') if isinstance(from_dev, dict) else None)
                            )
                            
                            out_interface = (
                                hop.get('outInterface') or 
                                hop.get('outIntf') or 
                                hop.get('outputInterface') or
                                hop.get('toIntf') or
                                hop.get('outboundInterface') or
                                (to_dev.get('interface') if isinstance(to_dev, dict) else None)
                            )
                            
                            # Check branch level for interface information
                            if not in_interface:
                                in_interface = branch.get('inInterface') or branch.get('inIntf') or branch.get('inputInterface')
                            if not out_interface:
                                out_interface = branch.get('outInterface') or branch.get('outIntf') or branch.get('outputInterface')
                        
                        if from_dev_name != 'Unknown' or to_dev_name:
                            hop_info = {
                                'from_device': from_dev_name,
                                'to_device': to_dev_name,
                                'status': branch_status,
                                'failure_reason': branch_failure_reason
                            }
                            
                            # Add firewall interface information if device is a firewall
                            if is_from_firewall or is_to_firewall:
                                if in_interface:
                                    hop_info['in_interface'] = in_interface
                                if out_interface:
                                    hop_info['out_interface'] = out_interface
                                hop_info['is_firewall'] = True
                            
                            simplified_hops.append(hop_info)
        
        return simplified_hops if simplified_hops else None
    except Exception as e:
        print(f"DEBUG: Error extracting hops from path_details: {e}", file=sys.stderr, flush=True)
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return None

def display_path_allowed_result(result, container):
    """
    Display path allowed/denied check result in a user-friendly format.
    Includes map visualization (same as path query) and firewall hostname when traffic is denied.
    
    Args:
        result: Dictionary containing the check result from check_path_allowed tool
        container: Streamlit container to display results in
    """
    if not result:
        container.error("❌ No result received from path allowed check.")
        return
    
    if "error" in result:
        container.error(f"❌ Error: {result['error']}")
        return
    
    # Extract key information
    source = result.get("source", "Unknown")
    destination = result.get("destination", "Unknown")
    protocol = result.get("protocol", "Unknown")
    port = result.get("port", "Unknown")
    status = result.get("status", "unknown")
    reason = result.get("reason", "No reason provided")
    path_exists = result.get("path_exists", False)
    path_hops = result.get("path_hops", [])
    firewall_denied_by = result.get("firewall_denied_by")
    
    # Display main status
    if status == "allowed":
        container.success(f"✅ **Traffic is ALLOWED**")
        container.info(f"**Path:** {source} → {destination} on {protocol}/{port}")
        container.info(f"**Reason:** {reason}")
        if path_exists:
            path_hops_count = result.get("path_hops_count", 0)
            if path_hops_count > 0:
                container.info(f"**Path Hops:** {path_hops_count} hops found")
    elif status == "denied":
        container.error(f"❌ **Traffic is DENIED**")
        container.info(f"**Path:** {source} → {destination} on {protocol}/{port}")
        container.info(f"**Reason:** {reason}")
        if firewall_denied_by:
            container.warning(f"**Firewall that denied traffic:** {firewall_denied_by}")
        policy_details = result.get("policy_details")
        if policy_details:
            container.warning(f"**Policy Details:** {policy_details}")
    else:
        container.warning(f"⚠️ **Status: UNKNOWN**")
        container.info(f"**Path:** {source} → {destination} on {protocol}/{port}")
        container.info(f"**Reason:** {reason}")
    
    # Map visualization (same as path query) when path_hops are available
    if path_hops and len(path_hops) > 0:
        try:
            graph_fig = create_path_graph(path_hops, source, destination)
            if graph_fig:
                container.markdown("### Network Path Graph")
                container.pyplot(graph_fig)
                plt.close(graph_fig)
        except Exception as e:
            container.warning(f"Could not generate path graph: {str(e)}")
    
    # Show additional details if available (omit internal status code)
    status_description = result.get("status_description")
    if status_description and status_description != "Success.":
        container.text(f"Status: {status_description}")


def display_result_chat(result, container):
    """
    Display graph visualization and firewall information from network path query result.
    
    Args:
        result: Dictionary containing the query result from the MCP server
        container: Streamlit container to display results in
    """
    import sys
    print(f"DEBUG: display_result_chat called with result type: {type(result)}", file=sys.stderr, flush=True)
    
    if not isinstance(result, dict):
        print(f"DEBUG: Result is not a dict, returning", file=sys.stderr, flush=True)
        return
    
    print(f"DEBUG: Result keys: {list(result.keys())}", file=sys.stderr, flush=True)
    
    # Check if this is a Panorama result (not a path query) - exit early
    if "ip_address" in result and ("address_objects" in result or "address_groups" in result or "device_group" in result or "vsys" in result or "error" in result):
        print(f"DEBUG: Skipping path display - this is a Panorama result, not a path query", file=sys.stderr, flush=True)
        print(f"DEBUG: Panorama result keys: {list(result.keys())}", file=sys.stderr, flush=True)
        # Display Panorama result properly instead
        if "ai_analysis" in result:
            ai_analysis = result["ai_analysis"]
            if isinstance(ai_analysis, dict):
                summary = ai_analysis.get("summary")
                if summary:
                    container.markdown(summary)
            elif isinstance(ai_analysis, str):
                container.markdown(ai_analysis)
        elif "error" in result:
            container.error(f"❌ {result['error']}")
        else:
            container.info("Query completed. No results found or analysis unavailable.")
        return
    
    # Debug: Print device cache info if available
    if "_debug_device_cache_size" in result:
        cache_size = result.get('_debug_device_cache_size', 0)
        cache_sample = result.get('_debug_device_cache_sample', [])
        print(f"DEBUG: Device Cache Debug - Size: {cache_size}, Sample: {cache_sample}", file=sys.stderr, flush=True)
    
    # Check for path hops in multiple possible locations FIRST
    hops_to_display = None
    if 'path_hops' in result and result['path_hops']:
        hops_to_display = result['path_hops']
        print(f"DEBUG: Using path_hops, count: {len(hops_to_display)}", file=sys.stderr, flush=True)
        # Debug: Print first hop to see device types
        if hops_to_display and len(hops_to_display) > 0:
            first_hop = hops_to_display[0]
            print(f"DEBUG: First hop keys: {list(first_hop.keys())}", file=sys.stderr, flush=True)
            print(f"DEBUG: First hop device types: from_device_type='{first_hop.get('from_device_type')}', to_device_type='{first_hop.get('to_device_type')}'", file=sys.stderr, flush=True)
    elif 'simplified_hops' in result and result['simplified_hops']:
        hops_to_display = result['simplified_hops']
        print(f"DEBUG: Using simplified_hops, count: {len(hops_to_display)}", file=sys.stderr, flush=True)
    elif 'path_details' in result and result['path_details']:
        print(f"DEBUG: Attempting to extract hops from path_details", file=sys.stderr, flush=True)
        hops_to_display = extract_hops_from_path_details(result['path_details'])
        if hops_to_display:
            print(f"DEBUG: Extracted {len(hops_to_display)} hops from path_details", file=sys.stderr, flush=True)
        else:
            print(f"DEBUG: Could not extract hops from path_details", file=sys.stderr, flush=True)
    
    # Display error information if no path data is available
    if not hops_to_display:
        path_failure_reason = result.get('path_failure_reason', '')
        path_status = result.get('path_status', '')
        path_status_description = result.get('path_status_description', '')
        
        print(f"DEBUG: No path hops found in result", file=sys.stderr, flush=True)
        
        # Display failure information if available
        if path_failure_reason or path_status == 'Failed' or 'Failed' in str(path_status):
            container.error("❌ Network Path Calculation Failed")
            
            if path_status_description:
                container.error(f"**Path Status:** {path_status_description}")
            elif path_status and path_status != 'Unknown':
                container.error(f"**Path Status:** {path_status}")
            
            if path_failure_reason:
                container.error(f"**Failure Reason:** {path_failure_reason}")
            
            # Show additional details if available
            if result.get('statusDescription'):
                container.info(f"Status: {result.get('statusDescription')}")
            if result.get('gateway_used'):
                container.info(f"Gateway Used: {result.get('gateway_used')}")
            if result.get('path_hops_count'):
                container.info(f"Hops Found: {result.get('path_hops_count')}")
        
        # If result only has error, display it
        elif "error" in result and len(result) == 1:
            container.error(f"❌ {result['error']}")
        else:
            # Display basic result information even if no hops
            container.warning("⚠️ Path query completed but no path data available to display.")
            if result.get('statusDescription'):
                container.info(f"Status: {result.get('statusDescription')}")
        
        return
    
    # Display graph visualization of the full path if hops are available
    if hops_to_display:
        print(f"DEBUG: Displaying graph with {len(hops_to_display)} hops", file=sys.stderr, flush=True)
        firewalls_found = {}
        for hop in hops_to_display:
            is_firewall = hop.get('is_firewall', False)
            if is_firewall:
                firewall_device = hop.get('firewall_device')
                if not firewall_device:
                    # Fallback: determine firewall device name
                    from_dev = hop.get('from_device', '')
                    to_dev = hop.get('to_device', '')
                    if 'fw' in from_dev.lower() or 'palo' in from_dev.lower() or 'fortinet' in from_dev.lower():
                        firewall_device = from_dev
                    elif to_dev and ('fw' in to_dev.lower() or 'palo' in to_dev.lower() or 'fortinet' in to_dev.lower()):
                        firewall_device = to_dev
                
                if firewall_device and firewall_device not in firewalls_found:
                    in_interface = hop.get('in_interface')
                    out_interface = hop.get('out_interface')
                    in_zone = hop.get('in_zone')
                    out_zone = hop.get('out_zone')
                    device_group = hop.get('device_group')
                    
                    # Extract device type - check both from_device and to_device to find the one matching firewall_device
                    device_type = None
                    from_dev = hop.get('from_device', '')
                    to_dev = hop.get('to_device', '')
                    if from_dev == firewall_device:
                        device_type = hop.get('from_device_type')
                    elif to_dev == firewall_device:
                        device_type = hop.get('to_device_type')
                    # Fallback: if device type not found, try to get it from either field
                    if not device_type:
                        device_type = hop.get('from_device_type') or hop.get('to_device_type')
                    
                    # Extract interface names (handle both string and dict formats)
                    in_intf_name = extract_interface_name(in_interface)
                    out_intf_name = extract_interface_name(out_interface)
                    
                    # Server logic: When firewall is "from", in_interface (or out_interface) is the firewall's OUT
                    # When firewall is "to", out_interface is the firewall's IN
                    if from_dev == firewall_device:
                        # Firewall is "from" - prefer API out_interface; if missing/same, infer egress (1/1 <-> 1/2) to match port-check
                        if not out_intf_name:
                            out_intf_name = extract_interface_name(hop.get("out_interface")) or infer_egress_interface(in_intf_name)
                        if in_intf_name and not out_intf_name:
                            out_intf_name = infer_egress_interface(in_intf_name) or in_intf_name
                            if out_intf_name != in_intf_name:
                                in_intf_name = in_intf_name  # keep in as ingress
                            else:
                                in_intf_name = None
                        elif in_intf_name and out_intf_name and _normalize_interface_for_compare(in_intf_name) == _normalize_interface_for_compare(out_intf_name):
                            out_intf_name = infer_egress_interface(in_intf_name) or out_intf_name  # infer distinct egress
                        if out_intf_name:
                            print(f"DEBUG: Table - Firewall {firewall_device} is 'from' device, out_interface: {out_intf_name}", file=sys.stderr, flush=True)
                    elif to_dev == firewall_device:
                        # Firewall is "to" device - out_interface from hop is the firewall's IN
                        if out_intf_name and not in_intf_name:
                            in_intf_name = out_intf_name
                            out_intf_name = None
                            print(f"DEBUG: Table - Firewall {firewall_device} is 'to' device, mapping out_interface to in_interface: {in_intf_name}", file=sys.stderr, flush=True)
                    
                    # Don't use same interface for both in and out - infer egress (1/1 <-> 1/2) to match port-check
                    if in_intf_name and out_intf_name and _normalize_interface_for_compare(in_intf_name) == _normalize_interface_for_compare(out_intf_name):
                        out_intf_name = infer_egress_interface(in_intf_name) or None
                    firewalls_found[firewall_device] = {
                        'in_interface': in_intf_name,
                        'out_interface': out_intf_name,
                        'in_zone': in_zone,
                        'out_zone': out_zone,
                        'device_group': device_group,
                        'device_type': device_type
                    }
                elif firewall_device in firewalls_found:
                    # Merge interface information if we have partial data
                    in_interface = hop.get('in_interface')
                    out_interface = hop.get('out_interface')
                    in_zone = hop.get('in_zone')
                    out_zone = hop.get('out_zone')
                    device_group = hop.get('device_group')
                    in_intf_name = extract_interface_name(in_interface)
                    out_intf_name = extract_interface_name(out_interface)
                    
                    # Server logic: When firewall is "from", in_interface is actually the firewall's OUT interface
                    # When firewall is "to", out_interface is actually the firewall's IN interface
                    # Match the server's mapping logic (same as graph code)
                    from_dev = hop.get('from_device', '')
                    to_dev = hop.get('to_device', '')
                    if from_dev == firewall_device:
                        # Firewall is "from" - infer egress when missing so table matches port-check
                        if in_intf_name and not out_intf_name:
                            out_intf_name = infer_egress_interface(in_intf_name) or in_intf_name
                            if out_intf_name != in_intf_name:
                                print(f"DEBUG: Table merge - Firewall {firewall_device} is 'from' device, inferred out_interface: {out_intf_name}", file=sys.stderr, flush=True)
                            else:
                                in_intf_name = None
                                print(f"DEBUG: Table merge - Firewall {firewall_device} is 'from' device, mapping in_interface to out_interface: {out_intf_name}", file=sys.stderr, flush=True)
                    elif to_dev == firewall_device:
                        # Firewall is "to" device - out_interface is actually the IN interface
                        if out_intf_name and not in_intf_name:
                            in_intf_name = out_intf_name
                            out_intf_name = None  # Don't use out_interface for OUT when firewall is "to"
                            print(f"DEBUG: Table merge - Firewall {firewall_device} is 'to' device, mapping out_interface to in_interface: {in_intf_name}", file=sys.stderr, flush=True)
                    
                    # Extract device type if not already set
                    if not firewalls_found[firewall_device].get('device_type'):
                        if from_dev == firewall_device:
                            device_type = hop.get('from_device_type')
                        elif to_dev == firewall_device:
                            device_type = hop.get('to_device_type')
                        else:
                            device_type = hop.get('from_device_type') or hop.get('to_device_type')
                        if device_type:
                            firewalls_found[firewall_device]['device_type'] = device_type
                    
                    # Update interfaces if we have new information (same logic for both)
                    if in_intf_name:
                        # Only update if we don't have a value yet
                        if not firewalls_found[firewall_device]['in_interface']:
                            firewalls_found[firewall_device]['in_interface'] = in_intf_name
                            print(f"DEBUG: Table - Set in_interface for {firewall_device} to {in_intf_name}", file=sys.stderr, flush=True)
                    if out_intf_name:
                        # Only update if we don't have a value yet and it's not the same as in_interface
                        existing_in = firewalls_found[firewall_device].get('in_interface')
                        if not firewalls_found[firewall_device]['out_interface']:
                            if not existing_in or _normalize_interface_for_compare(existing_in) != _normalize_interface_for_compare(out_intf_name):
                                firewalls_found[firewall_device]['out_interface'] = out_intf_name
                                print(f"DEBUG: Table - Set out_interface for {firewall_device} to {out_intf_name}", file=sys.stderr, flush=True)
                            else:
                                # Infer egress when same as in so table matches port-check
                                inferred_out = infer_egress_interface(existing_in)
                                if inferred_out:
                                    firewalls_found[firewall_device]['out_interface'] = inferred_out
                                    print(f"DEBUG: Table - Set out_interface for {firewall_device} to inferred {inferred_out}", file=sys.stderr, flush=True)
                                else:
                                    print(f"DEBUG: Table - Skipping out_interface for {firewall_device} (same as in_interface: {out_intf_name})", file=sys.stderr, flush=True)
                    else:
                        # When out_interface missing, try inferring from in_interface (match port-check)
                        existing_in = firewalls_found[firewall_device].get('in_interface')
                        if existing_in and not firewalls_found[firewall_device]['out_interface']:
                            inferred_out = infer_egress_interface(existing_in)
                            if inferred_out:
                                firewalls_found[firewall_device]['out_interface'] = inferred_out
                                print(f"DEBUG: Table - Set out_interface for {firewall_device} to inferred {inferred_out}", file=sys.stderr, flush=True)
                        if not firewalls_found[firewall_device].get('out_interface'):
                            print(f"DEBUG: Table - Failed to extract out_interface for {firewall_device}. Raw value: {out_interface}", file=sys.stderr, flush=True)
                    # Always update zones if available (they might come from different hops)
                    if in_zone:
                        firewalls_found[firewall_device]['in_zone'] = in_zone
                    if out_zone:
                        firewalls_found[firewall_device]['out_zone'] = out_zone
                    # Always update device group if available
                    if device_group:
                        firewalls_found[firewall_device]['device_group'] = device_group
        
        # Debug: Show device cache info if available (before graph)
        if "_debug_device_cache_size" in result:
            cache_size = result.get('_debug_device_cache_size', 0)
            cache_sample = result.get('_debug_device_cache_sample', [])
            print(f"DEBUG: Device Cache Info - Size: {cache_size}, Sample: {cache_sample}", file=sys.stderr, flush=True)
            # Print Devices API debug info if available
            if "_debug_devices_api" in result:
                api_debug = result["_debug_devices_api"]
                print(f"DEBUG: Devices API Debug - Endpoint: {api_debug.get('endpoint', 'N/A')}, Status: {api_debug.get('status', 'N/A')}, Devices: {api_debug.get('devices_count', 0)}, Cache Built: {api_debug.get('cache_built', False)}, Error: {api_debug.get('error', 'None')}", file=sys.stderr, flush=True)
            if cache_size > 0:
                container.success(f"✅ Device Cache: {cache_size} entries. Sample devices: {', '.join(cache_sample[:5])}")
            else:
                container.warning(f"⚠️ Device Cache: {cache_size} entries (cache is empty - device types will show as numbers)")
                # Show Devices API debug info if available
                with container.expander("🔍 Device Cache Debug Info"):
                    st.write(f"**Cache Size:** {cache_size}")
                    st.write(f"**Cache Sample:** {cache_sample}")
                    # Show Devices API debug info if available
                    if "_debug_devices_api" in result:
                        api_debug = result["_debug_devices_api"]
                        st.markdown("---")
                        st.markdown("**Devices API Debug Info:**")
                        st.write(f"**Endpoint:** `{api_debug.get('endpoint', 'N/A')}`")
                        st.write(f"**HTTP Status:** {api_debug.get('status', 'N/A')}")
                        st.write(f"**Devices Count:** {api_debug.get('devices_count', 0)}")
                        st.write(f"**Cache Built:** {api_debug.get('cache_built', False)}")
                        if api_debug.get('cache_size'):
                            st.write(f"**Cache Size:** {api_debug.get('cache_size', 0)}")
                        if api_debug.get('retry_status'):
                            st.write(f"**Retry Status:** {api_debug.get('retry_status')}")
                        if api_debug.get('error'):
                            st.error(f"**Error:** {api_debug['error']}")
                    else:
                        st.info("The Devices API call may have failed or returned no devices. Check server logs for details.")
        
        # Display graph visualization of the full path
        try:
            print(f"DEBUG: Creating graph with {len(hops_to_display)} hops", file=sys.stderr, flush=True)
            graph_fig = create_path_graph(
                hops_to_display,
                result.get('source', 'Source'),
                result.get('destination', 'Destination')
            )
            if graph_fig:
                print(f"DEBUG: Graph created successfully, displaying", file=sys.stderr, flush=True)
                container.markdown("#### Network Path Visualization")
                container.pyplot(graph_fig)
                import matplotlib.pyplot as plt
                plt.close(graph_fig)
            else:
                print(f"DEBUG: Graph creation returned None", file=sys.stderr, flush=True)
        except Exception as e:
            # Log error but don't fail silently
            import sys
            import traceback
            print(f"DEBUG: Graph generation error: {str(e)}", file=sys.stderr, flush=True)
            print(f"DEBUG: Graph traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
            container.warning(f"⚠️ Could not generate path visualization: {str(e)}")
        
        # Display firewall information in table format
        if firewalls_found:
            print(f"DEBUG: Found {len(firewalls_found)} firewalls", file=sys.stderr, flush=True)
            container.markdown("#### Firewalls in Path")
            
            # Prepare table data
            table_data = []
            for fw_name, fw_info in firewalls_found.items():
                # Use stored interface names directly (already extracted when building firewalls_found)
                # Same logic for both source and destination - just use the stored values
                source_intf = fw_info.get('in_interface')  # Already extracted
                dest_intf = fw_info.get('out_interface')   # Already extracted
                # Never show same interface for both source and destination - use inferred egress (match port-check)
                if source_intf and dest_intf and _normalize_interface_for_compare(source_intf) == _normalize_interface_for_compare(dest_intf):
                    dest_intf = infer_egress_interface(source_intf) or None
                if source_intf and not dest_intf:
                    dest_intf = infer_egress_interface(source_intf)
                # Get zones: source zone = zone for source interface (e.g. 1/1 → inside), destination zone = zone for dest interface (e.g. 1/2 → outside)
                in_zone = fw_info.get('in_zone')
                out_zone = fw_info.get('out_zone')
                device_group = fw_info.get('device_group', '')
                source_zone_display = (in_zone or '-').strip() if in_zone else '-'
                dest_zone_display = (out_zone or '-').strip() if out_zone else '-'
                # When destination was inferred and out_zone missing, infer zone (inside <-> outside) to match port-check
                if dest_intf and not out_zone and in_zone:
                    in_zl = (in_zone or "").strip().lower()
                    if in_zl == "inside":
                        dest_zone_display = "outside"
                    elif in_zl == "outside":
                        dest_zone_display = "inside"
                dg_display = (device_group or '-').strip() if device_group else '-'
                
                # Add row to table (column order: Name, Source Interface, Source Zone, Destination Interface, Destination Zone, Device Group)
                table_data.append({
                    'Name': fw_name,
                    'Source Interface': source_intf or '-',
                    'Source Zone': source_zone_display,
                    'Destination Interface': dest_intf or '-',
                    'Destination Zone': dest_zone_display,
                    'Device Group': dg_display
                })
            
            # Display as table using Streamlit's dataframe display
            if table_data:
                import pandas as pd
                df = pd.DataFrame(table_data)
                container.dataframe(df, width="stretch", hide_index=True)
        else:
            print(f"DEBUG: No firewalls found in path", file=sys.stderr, flush=True)
    else:
        print(f"DEBUG: No hops to display - result may not have path data", file=sys.stderr, flush=True)
        print(f"DEBUG: Result keys available: {list(result.keys())}", file=sys.stderr, flush=True)

def fetch_rack_elevation_image(elevation_url):
    """
    Fetch rack elevation image from NetBox URL.
    
    Note: NetBox elevation pages are HTML and may require authentication.
    This function tries multiple approaches to get the image.
    
    Args:
        elevation_url: URL to NetBox rack elevation page
        
    Returns:
        PIL Image object, image URL string, or None if fetch fails
    """
    if not elevation_url or not IMAGE_FETCH_AVAILABLE:
        return None
    
    try:
        # NetBox elevation URLs are HTML pages, not direct images
        # Try multiple approaches:
        
        # Approach 1: Try to construct direct image/render URLs
        # NetBox might have render endpoints or export functionality
        base_url = elevation_url.split('?')[0].rstrip('/')  # Remove query params and trailing slash
        image_urls_to_try = [
            base_url + '/render.png',
            base_url + '/render.svg',
            base_url + '/elevation.png',
            base_url + '/elevation.svg',
            base_url + '.png',
            base_url + '.svg',
            # Try with render parameter
            base_url + '?render=1',
            base_url + '?format=png',
            base_url + '?export=png',
        ]
        
        # Try fetching with authentication (if we had the token, but we don't in client)
        # For now, try without auth first
        for image_url in image_urls_to_try:
            try:
                response = requests.get(image_url, timeout=5, verify=False, allow_redirects=True)
                
                # Check if it's actually an image
                content_type = response.headers.get('content-type', '').lower()
                if content_type.startswith('image/'):
                    img = Image.open(BytesIO(response.content))
                    print(f"DEBUG: Successfully fetched elevation image from {image_url}", file=sys.stderr, flush=True)
                    return img
            except Exception as e:
                # Try next URL
                continue
        
        # Approach 2: Try to fetch HTML and extract image data
        # NetBox elevation pages might have embedded SVG or canvas data
        try:
            response = requests.get(elevation_url, timeout=5, verify=False, allow_redirects=True)
            if response.status_code == 200:
                html_content = response.text
                # Look for embedded images or SVG data in the HTML
                # This is a fallback - NetBox might render elevations as SVG
                import re
                # Try to find SVG content
                svg_match = re.search(r'<svg[^>]*>.*?</svg>', html_content, re.DOTALL | re.IGNORECASE)
                if svg_match:
                    # Found SVG - could convert to image, but for now return URL
                    # Streamlit can display SVG directly via URL
                    return elevation_url
        except Exception as e:
            pass
        
        # If all attempts failed, return the original URL
        # Streamlit might be able to display it directly, or user can click the link
        print(f"DEBUG: Could not fetch elevation image, returning URL: {elevation_url}", file=sys.stderr, flush=True)
        return elevation_url  # Return URL string instead of None
    except Exception as e:
        print(f"DEBUG: Error fetching elevation image from {elevation_url}: {e}", file=sys.stderr, flush=True)
        return elevation_url  # Return URL as fallback


def display_rack_details_result(result, container, format_type=None):
    """
    Display rack details lookup result.

    Args:
        result: Dictionary containing rack details data
        container: Streamlit container to display results in
        format_type: Optional format request ("table", "json", "list")
    """
    if not isinstance(result, dict):
        container.text(str(result))
        return

    if "error" in result:
        container.error(f"❌ {result['error']}")
        if "details" in result:
            container.info(f"Details: {result['details']}")
        return

    rack = result.get("rack", "Unknown rack")
    site = result.get("site")
    location = result.get("location")
    facility_id = result.get("facility_id")
    status = result.get("status")
    role = result.get("role")
    rack_type = result.get("type")
    width = result.get("width")
    u_height = result.get("u_height")
    devices_count = result.get("devices_count", 0)
    devices = result.get("devices", [])

    # Handle different format types
    if format_type == "table":
        import pandas as pd
        table_data = []
        table_data.append({"Field": "Rack", "Value": rack})
        if site:
            table_data.append({"Field": "Site", "Value": site})
        if location:
            table_data.append({"Field": "Location", "Value": location})
        if facility_id:
            table_data.append({"Field": "Facility ID", "Value": facility_id})
        if status:
            table_data.append({"Field": "Status", "Value": status})
        if role:
            table_data.append({"Field": "Role", "Value": role})
        if rack_type:
            table_data.append({"Field": "Type", "Value": rack_type})
        if width:
            table_data.append({"Field": "Width", "Value": width})
        if u_height:
            table_data.append({"Field": "Height (U)", "Value": u_height})
        space_utilization = result.get("space_utilization")
        if space_utilization is not None:
            table_data.append({"Field": "Space Utilization", "Value": f"{space_utilization}%"})
        occupied_units = result.get("occupied_units", 0)
        if occupied_units is not None and u_height:
            table_data.append({"Field": "Occupied Units", "Value": f"{occupied_units}U / {u_height}U"})
        table_data.append({"Field": "Devices Count", "Value": devices_count})
        
        df = pd.DataFrame(table_data)
        container.success(f"📍 {rack} - Rack Details")
        container.dataframe(df, width="stretch", hide_index=True)
        
        # Display devices in rack if available
        if devices and len(devices) > 0:
            container.markdown("#### Devices in Rack")
            devices_data = []
            for device in devices:
                devices_data.append({
                    "Device": device.get("name", "Unknown"),
                    "Position": f"U{int(device.get('position', 0))}" if device.get("position") else "N/A",
                    "Face": device.get("face", "N/A"),
                    "Type": device.get("device_type", "N/A"),
                    "Status": device.get("status", "N/A")
                })
            devices_df = pd.DataFrame(devices_data)
            container.dataframe(devices_df, width="stretch", hide_index=True)
        
    elif format_type == "json":
        container.json(result)
    elif format_type == "list":
        container.markdown(f"### {rack} - Rack Details")
        info_list = []
        if site:
            info_list.append(f"- **Site:** {site}")
        if location:
            info_list.append(f"- **Location:** {location}")
        if facility_id:
            info_list.append(f"- **Facility ID:** {facility_id}")
        if status:
            info_list.append(f"- **Status:** {status}")
        if role:
            info_list.append(f"- **Role:** {role}")
        if rack_type:
            info_list.append(f"- **Type:** {rack_type}")
        if width:
            info_list.append(f"- **Width:** {width}")
        if u_height:
            info_list.append(f"- **Height:** {u_height}U")
        info_list.append(f"- **Devices:** {devices_count}")
        container.markdown("\n".join(info_list))
        
        if devices and len(devices) > 0:
            container.markdown("#### Devices in Rack:")
            for device in devices:
                pos = f"U{int(device.get('position', 0))}" if device.get("position") else "N/A"
                container.markdown(f"- {device.get('name', 'Unknown')} (Position: {pos}, Face: {device.get('face', 'N/A')}, Type: {device.get('device_type', 'N/A')})")
    else:
        # Default: show summary with AI analysis if available
        container.markdown(f"### {rack} - Rack Details")
        location_info = []
        if site:
            location_info.append(f"**Site:** {site}")
        if location:
            location_info.append(f"**Location:** {location}")
        if facility_id:
            location_info.append(f"**Facility ID:** {facility_id}")
        if status:
            location_info.append(f"**Status:** {status}")
        if u_height:
            location_info.append(f"**Height:** {u_height}U")
        location_info.append(f"**Devices:** {devices_count}")
        
        if location_info:
            container.markdown("\n".join(location_info))
        
        # Display AI analysis if available
        if "ai_analysis" in result:
            ai_analysis = result["ai_analysis"]
            if isinstance(ai_analysis, dict):
                summary = ai_analysis.get("summary")
                if summary:
                    container.markdown(summary)
            elif isinstance(ai_analysis, str):
                container.markdown(ai_analysis)


def display_racks_list_result(result, container, format_type=None):
    """
    Display list of racks result.
    
    Args:
        result: Dictionary containing racks list data
        container: Streamlit container to display results in
        format_type: Optional format request ("table", "json", "list")
    """
    if not isinstance(result, dict):
        container.text(str(result))
        return
    
    if "error" in result:
        container.error(f"❌ {result['error']}")
        if "details" in result:
            container.info(f"Details: {result['details']}")
        return
    
    racks = result.get("racks", [])
    total_count = result.get("total_count", len(racks))
    site_filter = result.get("site_filter")
    
    if format_type == "table":
        import pandas as pd
        if racks:
            racks_data = []
            for rack in racks:
                racks_data.append({
                    "Rack": rack.get("rack", "Unknown"),
                    "Site": rack.get("site", "N/A"),
                    "Status": rack.get("status", "N/A"),
                    "Height (U)": rack.get("u_height", "N/A"),
                    "Space Utilization": f"{rack.get('space_utilization', 0)}%" if rack.get("space_utilization") is not None else "N/A",
                    "Occupied Units": f"{rack.get('occupied_units', 0)}U" if rack.get("occupied_units") is not None else "N/A",
                    "Devices": rack.get("devices_count", 0)
                })
            
            df = pd.DataFrame(racks_data)
            title = f"📍 All Racks"
            if site_filter:
                title = f"📍 Racks at {site_filter}"
            container.success(f"{title} ({total_count} total)")
            container.dataframe(df, width="stretch", hide_index=True)
        else:
            container.info(f"No racks found{' at ' + site_filter if site_filter else ''}.")
    elif format_type == "json":
        container.json(result)
    elif format_type == "list":
        title = "All Racks"
        if site_filter:
            title = f"Racks at {site_filter}"
        container.markdown(f"### {title} ({total_count} total)")
        if racks:
            for rack in racks:
                container.markdown(f"- **{rack.get('rack', 'Unknown')}** at {rack.get('site', 'Unknown site')} - {rack.get('space_utilization', 0)}% utilized, {rack.get('devices_count', 0)} devices")
        else:
            container.info(f"No racks found{' at ' + site_filter if site_filter else ''}.")
    else:
        # Default: show table format
        import pandas as pd
        if racks:
            racks_data = []
            for rack in racks:
                racks_data.append({
                    "Rack": rack.get("rack", "Unknown"),
                    "Site": rack.get("site", "N/A"),
                    "Status": rack.get("status", "N/A"),
                    "Height (U)": rack.get("u_height", "N/A"),
                    "Space Utilization": f"{rack.get('space_utilization', 0)}%" if rack.get("space_utilization") is not None else "N/A",
                    "Occupied Units": f"{rack.get('occupied_units', 0)}U" if rack.get("occupied_units") is not None else "N/A",
                    "Devices": rack.get("devices_count", 0)
                })
            
            df = pd.DataFrame(racks_data)
            title = f"📍 All Racks"
            if site_filter:
                title = f"📍 Racks at {site_filter}"
            container.success(f"{title} ({total_count} total)")
            container.dataframe(df, width="stretch", hide_index=True)
            
            # Show AI analysis if available
            ai_analysis = result.get("ai_analysis")
            if ai_analysis:
                if isinstance(ai_analysis, dict):
                    summary = ai_analysis.get("summary", "")
                    if summary:
                        container.markdown(f"**Summary:** {summary}")
                elif isinstance(ai_analysis, str):
                    container.markdown(ai_analysis)
        else:
            container.info(f"No racks found{' at ' + site_filter if site_filter else ''}.")


def display_rack_location_result(result, container, format_type=None, intent=None, yes_no_question=None):
    """
    Display rack location lookup result.

    Args:
        result: Dictionary containing rack location data
        container: Streamlit container to display results in
        format_type: Optional format request ("table", "json", "list", "minimal", "summary")
        intent: Optional intent ("rack_location_only", "device_details", "device_type_only", "status_only", "site_only", "manufacturer_only")
    """
    if not isinstance(result, dict):
        container.text(str(result))
        return

    if "error" in result:
        container.error(f"❌ {result['error']}")
        if "details" in result:
            container.info(f"Details: {result['details']}")
        return

    # Get intent from result if not provided as parameter (for re-rendering)
    if not intent and "intent" in result:
        intent = result.get("intent")
    
    # Also check intent_output for stored intent from previous renders
    if not intent and "intent_output" in result:
        intent = result.get("intent_output")
    
    print(f"DEBUG: display_rack_location_result - Final intent: {intent}, result keys: {list(result.keys())}", file=sys.stderr, flush=True)
    
    device = result.get("device", "Unknown device")
    rack = result.get("rack")
    position = result.get("position")
    site = result.get("site")
    status = result.get("status")

    # Only check for rack assignment if the intent requires rack information
    # Intents that DON'T need rack: manufacturer_only, device_type_only, status_only, site_only
    requires_rack = intent not in ("manufacturer_only", "device_type_only", "status_only", "site_only")
    
    if not rack and requires_rack:
        message = result.get("message") or "Device is not assigned to a rack."
        container.warning(f"⚠️ {device}: {message}")
        if site:
            container.info(f"Site: {site}")
        return

    # Format position as "U1" instead of "1.0"
    position_str = None
    if position is not None:
        try:
            position_int = int(float(position))
            position_str = f"U{position_int}"
        except (ValueError, TypeError):
            position_str = str(position) if position else None

    # If intent is for a specific field (site, status, device_type, manufacturer), always use table format
    # even if format_type is minimal/summary, because we need to show only that specific field
    if intent in ("site_only", "status_only", "device_type_only", "manufacturer_only"):
        format_type = "table"  # Override format to table for specific field requests

    # Handle minimal/summary format - show only rack location
    # BUT skip this if intent is for a specific field (handled above)
    if format_type in ("minimal", "summary") and intent not in ("site_only", "status_only", "device_type_only", "manufacturer_only"):
        container.markdown(f"### {device} - Rack Location")
        location_info = []
        if site:
            location_info.append(f"**Site:** {site}")
        if rack:
            location_info.append(f"**Rack:** {rack}")
        if position_str:
            location_info.append(f"**Position:** {position_str}")
        
        if location_info:
            container.markdown("\n".join(location_info))
        else:
            container.info("Rack location information not available.")
        return

    # Display in table format if requested
    if format_type == "table":
        import pandas as pd
        table_data = []
        
        print(f"DEBUG: display_rack_location_result - format_type: {format_type}, intent: {intent}", file=sys.stderr, flush=True)
        print(f"DEBUG: Checking intent: '{intent}' (type: {type(intent)}, repr: {repr(intent)})", file=sys.stderr, flush=True)
        
        # Normalize intent (strip whitespace, convert to string) - DO THIS FIRST
        if intent:
            intent = str(intent).strip()
            print(f"DEBUG: Normalized intent: '{intent}'", file=sys.stderr, flush=True)
        
        # Handle specific intents - show only requested fields
        # Check these BEFORE device_details to ensure specific intents are handled
        if intent == "rack_location_only":
            # Only show rack location fields
            if site:
                table_data.append({"Field": "Site", "Value": site})
            if rack:
                table_data.append({"Field": "Rack", "Value": rack})
            if position is not None:
                position_int = int(float(position)) if position else None
                if position_int is not None:
                    table_data.append({"Field": "Position", "Value": f"U{position_int}"})
            
            df = pd.DataFrame(table_data)
            container.success(f"📍 {device} - Rack Location")
            container.dataframe(df, width="stretch", hide_index=True)
            return  # CRITICAL: Return early to prevent showing other fields
        elif intent == "device_type_only":
            # Check if this is a yes/no question (check result first for re-rendering, then parameter, then session_state)
            yes_no_q = result.get("yes_no_question") or yes_no_question or st.session_state.get("yes_no_question")
            if yes_no_q and yes_no_q.get("question_field") == "device_type":
                expected_value = yes_no_q.get("expected_value", "").strip()
                actual_device_type = result.get("device_type", "").strip()
                # Compare (case-insensitive)
                is_match = actual_device_type.lower() == expected_value.lower()
                answer = "Yes" if is_match else "No"
                container.success(f"**{answer}**")
                return
            
            print(f"DEBUG: device_type_only intent detected! Showing only device type", file=sys.stderr, flush=True)
            # Only show device type (normal display)
            device_type = result.get("device_type")
            print(f"DEBUG: Device type value from result: {device_type}", file=sys.stderr, flush=True)
            if device_type:
                table_data.append({"Field": "Device Type", "Value": device_type})
            else:
                print(f"DEBUG: WARNING - No device_type in result! Result keys: {list(result.keys())}", file=sys.stderr, flush=True)
            df = pd.DataFrame(table_data)
            container.success(f"📍 {device} - Device Type")
            container.dataframe(df, width="stretch", hide_index=True)
            print(f"DEBUG: Returning early from device_type_only handler", file=sys.stderr, flush=True)
            return  # CRITICAL: Return early to prevent showing other fields
        elif intent == "status_only":
            # Check if this is a yes/no question (check result first for re-rendering, then parameter, then session_state)
            yes_no_q = result.get("yes_no_question") or yes_no_question or st.session_state.get("yes_no_question")
            if yes_no_q and yes_no_q.get("question_field") == "status":
                expected_value = yes_no_q.get("expected_value", "").strip()
                actual_status = result.get("status", "").strip()
                # Compare (case-insensitive)
                is_match = actual_status.lower() == expected_value.lower()
                answer = "Yes" if is_match else "No"
                container.success(f"**{answer}**")
                return
            
            # Only show status (normal display)
            if status:
                table_data.append({"Field": "Status", "Value": status})
            df = pd.DataFrame(table_data)
            container.success(f"📍 {device} - Status")
            container.dataframe(df, width="stretch", hide_index=True)
            return  # CRITICAL: Return early to prevent showing other fields
        elif intent == "manufacturer_only":
            # Check if this is a yes/no question (check result first for re-rendering, then parameter, then session_state)
            yes_no_q = result.get("yes_no_question") or yes_no_question or st.session_state.get("yes_no_question")
            if yes_no_q and yes_no_q.get("question_field") == "manufacturer":
                expected_value = yes_no_q.get("expected_value", "").strip()
                manufacturer = result.get("manufacturer", "").strip()
                # Compare (case-insensitive)
                is_match = manufacturer.lower() == expected_value.lower()
                answer = "Yes" if is_match else "No"
                container.success(f"**{answer}**")
                return
            
            # Only show manufacturer (normal display)
            manufacturer = result.get("manufacturer")
            if manufacturer:
                table_data.append({"Field": "Manufacturer", "Value": manufacturer})
            df = pd.DataFrame(table_data)
            container.success(f"📍 {device} - Manufacturer")
            container.dataframe(df, width="stretch", hide_index=True)
            return  # CRITICAL: Return early to prevent showing other fields
        elif intent == "site_only":
            # Only show site
            if site:
                table_data.append({"Field": "Site", "Value": site})
            df = pd.DataFrame(table_data)
            container.success(f"📍 {device} - Site")
            container.dataframe(df, width="stretch", hide_index=True)
            return  # CRITICAL: Return early to prevent showing other fields
        elif intent == "device_details" or intent is None:
            print(f"DEBUG: device_details or None intent - showing all fields. Intent was: {repr(intent)}", file=sys.stderr, flush=True)
            print(f"DEBUG: WARNING - This should not happen for device_type_only queries! Check intent extraction.", file=sys.stderr, flush=True)
            # Show all device details (default behavior)
            table_data.append({"Field": "Device", "Value": device})
            if rack:
                table_data.append({"Field": "Rack", "Value": rack})
            if position is not None:
                position_int = int(float(position)) if position else None
                if position_int is not None:
                    table_data.append({"Field": "Position", "Value": f"U{position_int}"})
            if site:
                table_data.append({"Field": "Site", "Value": site})
            if status:
                table_data.append({"Field": "Status", "Value": status})
            
            # Add device details from result if available
            if "device_type" in result:
                table_data.append({"Field": "Device Type", "Value": result.get("device_type")})
            if "device_role" in result:
                table_data.append({"Field": "Device Role", "Value": result.get("device_role")})
            if "manufacturer" in result:
                table_data.append({"Field": "Manufacturer", "Value": result.get("manufacturer")})
            if "model" in result:
                table_data.append({"Field": "Model", "Value": result.get("model")})
            if "serial" in result:
                table_data.append({"Field": "Serial", "Value": result.get("serial")})
            if "primary_ip" in result:
                table_data.append({"Field": "Primary IP", "Value": result.get("primary_ip")})
            if "primary_ip4" in result:
                table_data.append({"Field": "Primary IPv4", "Value": result.get("primary_ip4")})
            
            df = pd.DataFrame(table_data)
            container.success(f"📍 {device} - Device Details")
            container.dataframe(df, width="stretch", hide_index=True)
    else:
        # Only show location info if AI analysis is not present
        if "ai_analysis" not in result or not result.get("ai_analysis"):
            location_parts = [f"Rack: {rack}"]
            if position is not None:
                # Format position as U{number} (e.g., U1 for position 1.0)
                position_int = int(float(position)) if position else None
                if position_int is not None:
                    location_parts.append(f"Position: U{position_int}")
            if site:
                location_parts.append(f"Site: {site}")
            if status:
                location_parts.append(f"Status: {status}")

            container.success(f"📍 {device} location")
            container.markdown(", ".join(location_parts))
    
    # Display AI analysis if available (only if not in table format, or show it after the table)
    print(f"DEBUG: Client - Checking for AI analysis. Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}", file=sys.stderr, flush=True)
    print(f"DEBUG: Client - format_type: {format_type}", file=sys.stderr, flush=True)
    if "ai_analysis" in result:
        print(f"DEBUG: Client - AI analysis found in result, displaying...", file=sys.stderr, flush=True)
        ai_analysis = result["ai_analysis"]
        print(f"DEBUG: Client - AI analysis type: {type(ai_analysis)}, content: {str(ai_analysis)[:200]}", file=sys.stderr, flush=True)
        # If table format, don't show AI analysis (table is the primary display)
        # If not table format, show AI analysis as the summary
        if format_type != "table":
            if isinstance(ai_analysis, dict):
                summary = ai_analysis.get("summary") or ai_analysis.get("Summary") or ai_analysis.get("SUMMARY")
                if summary:
                    container.markdown(summary)
            elif isinstance(ai_analysis, str):
                container.markdown(ai_analysis)
    else:
        print(f"DEBUG: Client - No AI analysis in result. Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}", file=sys.stderr, flush=True)

async def execute_path_allowed_check(source, destination, protocol, port, is_live):
    """
    Execute path allowed/denied check asynchronously.
    
    Args:
        source: Source IP/hostname
        destination: Destination IP/hostname
        protocol: Protocol (TCP/UDP)
        port: Port number
        is_live: Use live data (True/False)
        
    Returns:
        dict: Policy check result
    """
    import sys
    print(f"DEBUG: Starting path allowed check: {source} -> {destination}, protocol={protocol}, port={port}, is_live={is_live}", file=sys.stderr, flush=True)
    
    try:
        print(f"DEBUG: Connecting to MCP server...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized, calling check_path_allowed tool...", file=sys.stderr, flush=True)
            
            # Use defaults if protocol or port are None (server expects strings, not None)
            protocol_str = protocol if protocol is not None else "TCP"
            port_str = port if port is not None else "0"
            
            tool_arguments = {
                "source": source,
                "destination": destination,
                "protocol": protocol_str,
                "port": port_str,
                "is_live": 1 if is_live else 0
            }
            print(f"DEBUG: Tool arguments: {tool_arguments}", file=sys.stderr, flush=True)
            
            # Try standard format first, then FastMCP format if needed
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    # FastMCPClient not available or not importable
                    pass
            
            try:
                # Try standard MCP format first
                result = await asyncio.wait_for(
                    client_or_session.call_tool("check_path_allowed", arguments=tool_arguments),
                    timeout=360.0  # 6 minute timeout for path checks
                )
                print(f"DEBUG: Standard format succeeded for check_path_allowed", file=sys.stderr, flush=True)
            except TypeError as e:
                if "unexpected keyword argument 'arguments'" in str(e) and is_fastmcp:
                    # FastMCP client expects **kwargs instead of arguments=dict
                    print(f"DEBUG: Standard format failed, trying FastMCP format (kwargs) for check_path_allowed", file=sys.stderr, flush=True)
                    result = await asyncio.wait_for(
                        client_or_session.call_tool("check_path_allowed", **tool_arguments),
                        timeout=360.0
                    )
                else:
                    raise
            
            # Process result based on client type
            # Check if result has content attribute (standard MCP format)
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    content_item = result.content[0]
                    result_text = content_item.text if hasattr(content_item, 'text') else str(content_item)
                else:
                    result_text = str(result.content)
            else:
                result_text = str(result)
            
            print(f"DEBUG: Result text length: {len(result_text)}", file=sys.stderr, flush=True)
            print(f"DEBUG: Result text (first 500 chars): {result_text[:500]}", file=sys.stderr, flush=True)
            
            try:
                result_dict = json.loads(result_text)
                return result_dict
            except json.JSONDecodeError:
                return {"result": result_text}
    
    except asyncio.TimeoutError:
        return {"error": "Path allowed check timed out. Please try again."}
    except Exception as e:
        import traceback
        print(f"DEBUG: Error in execute_path_allowed_check: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": f"Error executing path allowed check: {str(e)}"}


async def execute_network_query(source, destination, protocol, port, is_live):
    """
    Execute network path query asynchronously.
    
    Args:
        source: Source IP/hostname
        destination: Destination IP/hostname
        protocol: Protocol (TCP/UDP)
        port: Port number
        is_live: Use live data (True/False)
        
    Returns:
        dict: Query result
    """
    import sys
    print(f"DEBUG: Starting network query: {source} -> {destination}, protocol={protocol}, port={port}, is_live={is_live}", file=sys.stderr, flush=True)
    
    try:
        print(f"DEBUG: Connecting to MCP server...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized, calling tool...", file=sys.stderr, flush=True)
            
            # Use defaults if protocol or port are None (server expects strings, not None)
            protocol_str = protocol if protocol is not None else "TCP"
            port_str = port if port is not None else "0"
            
            tool_arguments = {
                "source": source,
                "destination": destination,
                "protocol": protocol_str,
                "port": port_str,
                "is_live": 1 if is_live else 0,
                "continue_on_policy_denial": True  # Always continue even if denied by policy
            }
            print(f"DEBUG: Tool arguments: {tool_arguments}", file=sys.stderr, flush=True)
            print(f"DEBUG: Calling tool with arguments: {tool_arguments}", file=sys.stderr, flush=True)
            print(f"DEBUG: Client type: {type(client_or_session).__name__}, module: {type(client_or_session).__module__}, FASTMCP_CLIENT_AVAILABLE: {FASTMCP_CLIENT_AVAILABLE}", file=sys.stderr, flush=True)
            try:
                # Try to detect FastMCP Client - be conservative, default to standard format
                is_fastmcp = False
                if FASTMCP_CLIENT_AVAILABLE:
                    try:
                        # Check if it's actually a FastMCPClient instance
                        is_fastmcp = isinstance(client_or_session, FastMCPClient)
                        print(f"DEBUG: isinstance check result: {is_fastmcp}", file=sys.stderr, flush=True)
                    except (NameError, TypeError) as e:
                        print(f"DEBUG: isinstance check failed: {e}, checking by module name", file=sys.stderr, flush=True)
                        # Fallback: check by module name (more reliable)
                        module_name = type(client_or_session).__module__
                        is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
                        print(f"DEBUG: Module-based check result: {is_fastmcp} (module: {module_name})", file=sys.stderr, flush=True)
                
                print(f"DEBUG: Final detection - is_fastmcp: {is_fastmcp}, type: {type(client_or_session).__name__}, module: {type(client_or_session).__module__}", file=sys.stderr, flush=True)
                
                # Try standard format first (safest), then FastMCP format if needed
                try:
                    # Standard MCP ClientSession format: pass arguments as a dictionary
                    print(f"DEBUG: Trying standard format (arguments=dict) for query_network_path...", file=sys.stderr, flush=True)
                    tool_result = await asyncio.wait_for(
                        client_or_session.call_tool("query_network_path", arguments=tool_arguments),
                        timeout=360.0  # 6 minute timeout for network path queries (server polls up to 120 times)
                    )
                    print(f"DEBUG: Standard format succeeded for query_network_path, result type: {type(tool_result)}", file=sys.stderr, flush=True)
                except TypeError as e:
                    error_str = str(e)
                    # If standard format fails with "unexpected keyword argument 'arguments'", try FastMCP format
                    if "unexpected keyword argument 'arguments'" in error_str and is_fastmcp:
                        print(f"DEBUG: Standard format failed, trying FastMCP format (keyword args) for query_network_path...", file=sys.stderr, flush=True)
                        tool_result_list = await asyncio.wait_for(
                            client_or_session.call_tool("query_network_path", **tool_arguments),
                            timeout=360.0  # 6 minute timeout for network path queries (server polls up to 120 times)
                        )
                        print(f"DEBUG: FastMCP call completed, result type: {type(tool_result_list)}, length: {len(tool_result_list) if isinstance(tool_result_list, list) else 'N/A'}", file=sys.stderr, flush=True)
                        # FastMCP returns a list of results, each with .text attribute
                        # Convert to standard format for processing
                        if tool_result_list and len(tool_result_list) > 0:
                            # Create a mock tool_result object with content attribute
                            class FastMCPToolResult:
                                def __init__(self, results):
                                    self.content = []
                                    for r in results:
                                        text = r.text if hasattr(r, 'text') else str(r)
                                        self.content.append(type('Content', (), {'text': text})())
                            tool_result = FastMCPToolResult(tool_result_list)
                            print(f"DEBUG: FastMCP result wrapped, content items: {len(tool_result.content)}", file=sys.stderr, flush=True)
                        else:
                            tool_result = None
                            print(f"DEBUG: FastMCP returned empty result", file=sys.stderr, flush=True)
                        print(f"DEBUG: FastMCP format succeeded for query_network_path", file=sys.stderr, flush=True)
                    else:
                        # Re-raise if it's a different error
                        print(f"DEBUG: Error calling query_network_path: {e}", file=sys.stderr, flush=True)
                        raise
                print(f"DEBUG: Tool call completed, processing result...", file=sys.stderr, flush=True)
            except asyncio.TimeoutError:
                print(f"DEBUG: Tool call timed out after 360 seconds", file=sys.stderr, flush=True)
                return {"error": "Network path query timed out after 6 minutes. The query may be too complex or the server may be slow. Please try again or use baseline data instead of live data."}
            except (ConnectionResetError, ConnectionError, OSError) as conn_error:
                # Connection was closed, but server may have completed - check if we got partial result
                print(f"DEBUG: Connection error during tool call: {conn_error}", file=sys.stderr, flush=True)
                print(f"DEBUG: Connection error type: {type(conn_error).__name__}", file=sys.stderr, flush=True)
                # If we have a tool_result, try to process it anyway
                if 'tool_result' in locals() and tool_result:
                    print(f"DEBUG: Connection closed but we have a result, attempting to process...", file=sys.stderr, flush=True)
                    # Fall through to result processing below
                else:
                    import traceback
                    print(f"DEBUG: Connection error traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                    return {"error": f"Connection was closed during query. The server may have completed processing, but the connection was lost. Error: {str(conn_error)}"}
            except Exception as tool_error:
                print(f"DEBUG: Tool call failed with error: {tool_error}", file=sys.stderr, flush=True)
                import traceback
                print(f"DEBUG: Tool call error traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                # If we have a tool_result despite the error, try to process it
                if 'tool_result' in locals() and tool_result:
                    print(f"DEBUG: Error occurred but we have a result, attempting to process...", file=sys.stderr, flush=True)
                    # Fall through to result processing below
                else:
                    return {"error": f"Tool call failed: {str(tool_error)}"}
            
            if tool_result:
                print(f"DEBUG: Tool result received, type: {type(tool_result)}, dir: {[x for x in dir(tool_result) if not x.startswith('_')]}", file=sys.stderr, flush=True)
                result_text = None
                
                # Handle FastMCP Client response (list of results or FastMCPToolResult wrapper)
                if isinstance(tool_result, list):
                    if len(tool_result) > 0:
                        # FastMCP returns list of result objects with .text attribute
                        first_result = tool_result[0]
                        if hasattr(first_result, 'text'):
                            result_text = first_result.text
                        else:
                            result_text = str(first_result)
                        print(f"DEBUG: FastMCP list result (first 500 chars): {result_text[:500] if result_text else 'None'}", file=sys.stderr, flush=True)
                    else:
                        return {"error": "Tool call returned empty result"}
                # Handle FastMCPToolResult wrapper or standard MCP ClientSession response (both have .content)
                elif hasattr(tool_result, 'content') and tool_result.content:
                    import json
                    print(f"DEBUG: Standard MCP result content length: {len(tool_result.content)}", file=sys.stderr, flush=True)
                    if isinstance(tool_result.content, list) and len(tool_result.content) > 0:
                        content_item = tool_result.content[0]
                        if hasattr(content_item, 'text'):
                            result_text = content_item.text
                        else:
                            result_text = str(content_item)
                    else:
                        result_text = str(tool_result.content)
                    print(f"DEBUG: Result text length: {len(result_text) if result_text else 0}", file=sys.stderr, flush=True)
                    print(f"DEBUG: Result text (first 500 chars): {result_text[:500] if result_text else 'None'}", file=sys.stderr, flush=True)
                else:
                    # Try to convert to string or check if it's already a dict
                    if isinstance(tool_result, dict):
                        print(f"DEBUG: Tool result is already a dict, returning directly", file=sys.stderr, flush=True)
                        return tool_result
                    result_text = str(tool_result)
                    print(f"DEBUG: Converted result to string (first 500 chars): {result_text[:500]}", file=sys.stderr, flush=True)
                
                if not result_text:
                    print(f"DEBUG: ERROR - result_text is None or empty", file=sys.stderr, flush=True)
                    return {"error": "Tool call returned empty or unparseable result"}
                
                # Try to parse as JSON
                import json
                try:
                    result = json.loads(result_text)
                    print(f"DEBUG: Result parsed successfully, keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}", file=sys.stderr, flush=True)
                    return result
                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON decode error: {e}, result_text: {result_text[:200]}", file=sys.stderr, flush=True)
                    # Try to extract JSON from the text if it's embedded
                    import re
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group())
                            print(f"DEBUG: Extracted JSON from text, keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}", file=sys.stderr, flush=True)
                            return result
                        except json.JSONDecodeError as e2:
                            print(f"DEBUG: Failed to parse extracted JSON: {e2}", file=sys.stderr, flush=True)
                    # If still can't parse, return the raw text wrapped
                    print(f"DEBUG: Returning raw text as result", file=sys.stderr, flush=True)
                    return {"error": f"Failed to parse JSON result: {str(e)}", "raw_result": result_text[:1000]}
            else:
                print(f"DEBUG: Tool result is None or empty", file=sys.stderr, flush=True)
                return {"error": "Tool call returned no result"}
    except asyncio.TimeoutError:
        print(f"DEBUG: Query timed out", file=sys.stderr, flush=True)
        return {"error": "Query timed out. The network path calculation is taking longer than expected."}
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"DEBUG: Exception in execute_network_query: {e}", file=sys.stderr, flush=True)
        print(f"DEBUG: Traceback: {error_traceback}", file=sys.stderr, flush=True)
        # Include more details in the error message
        error_msg = f"Error executing query: {str(e)}"
        if "JSON" in str(e) or "json" in str(e):
            error_msg += " (JSON parsing error - check server logs)"
        elif "TaskGroup" in str(e) or "unhandled errors" in str(e):
            error_msg += " (MCP protocol error - server may have crashed, check mcp_server.log)"
        return {"error": error_msg}


async def execute_rack_details_query(rack_name, format_type=None, conversation_history=None, site_name=None):
    """
    Execute rack details lookup via MCP server.

    Args:
        rack_name: Rack name to look up in NetBox
        format_type: Optional format request ("table", "json", "list")
        conversation_history: Optional conversation history for context
        site_name: Optional site name to filter racks

    Returns:
        dict: Rack details result
    """
    import sys
    print(f"DEBUG: Starting rack details query for rack: {rack_name}, site: {site_name}, format: {format_type}", file=sys.stderr, flush=True)

    try:
        print(f"DEBUG: Connecting to MCP server for rack details query...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized for rack details query", file=sys.stderr, flush=True)
            
            tool_arguments = {"rack_name": rack_name}
            if site_name:
                tool_arguments["site_name"] = site_name
            if format_type:
                tool_arguments["format"] = format_type
            if conversation_history:
                tool_arguments["conversation_history"] = conversation_history
            
            # Detect client type for fallback
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
            
            print(f"DEBUG: Client type for rack details: FastMCP={is_fastmcp}, type: {type(client_or_session).__name__}, module: {type(client_or_session).__module__}", file=sys.stderr, flush=True)
            
            # Try standard format first (safest), then FastMCP format if needed
            try:
                # Standard MCP ClientSession format: pass arguments as a dictionary
                print(f"DEBUG: Trying standard format (arguments=dict) for get_rack_details...", file=sys.stderr, flush=True)
                tool_result = await asyncio.wait_for(
                    client_or_session.call_tool("get_rack_details", arguments=tool_arguments),
                    timeout=60.0
                )
                print(f"DEBUG: Standard format succeeded for get_rack_details", file=sys.stderr, flush=True)
            except TypeError as e:
                error_str = str(e)
                # If standard format fails with "unexpected keyword argument 'arguments'", try FastMCP format
                if "unexpected keyword argument 'arguments'" in error_str and is_fastmcp:
                    print(f"DEBUG: Standard format failed, trying FastMCP format (keyword args) for get_rack_details...", file=sys.stderr, flush=True)
                    tool_result_list = await asyncio.wait_for(
                        client_or_session.call_tool("get_rack_details", **tool_arguments),
                        timeout=60.0
                    )
                    # Convert FastMCP result to standard format
                    if tool_result_list and len(tool_result_list) > 0:
                        class FastMCPToolResult:
                            def __init__(self, results):
                                self.content = []
                                for r in results:
                                    text = r.text if hasattr(r, 'text') else str(r)
                                    self.content.append(type('Content', (), {'text': text})())
                        tool_result = FastMCPToolResult(tool_result_list)
                        print(f"DEBUG: FastMCP format succeeded for get_rack_details", file=sys.stderr, flush=True)
                    else:
                        tool_result = None
                else:
                    # Re-raise if it's a different error
                    print(f"DEBUG: Error calling get_rack_details: {e}", file=sys.stderr, flush=True)
                    raise

            if tool_result and tool_result.content:
                import json
                result_text = tool_result.content[0].text
                print(f"DEBUG: Rack details result text length: {len(result_text)}", file=sys.stderr, flush=True)
                try:
                    result = json.loads(result_text)
                    print(f"DEBUG: Rack details result parsed successfully", file=sys.stderr, flush=True)
                    return result
                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON decode error: {e}", file=sys.stderr, flush=True)
                    return {"result": result_text}
            else:
                print(f"DEBUG: No result content returned", file=sys.stderr, flush=True)
                return None
    except asyncio.TimeoutError:
        return {"error": "Rack details lookup timed out"}
    except Exception as e:
        import sys
        print(f"DEBUG: Error executing rack details query: {str(e)}", file=sys.stderr, flush=True)
        return {"error": f"Error executing query: {str(e)}"}


async def execute_racks_list_query(site_name=None, format_type=None, conversation_history=None):
    """
    Execute racks list lookup via MCP server.
    
    Args:
        site_name: Optional site name to filter racks
        format_type: Optional format request ("table", "json", "list")
        conversation_history: Optional conversation history for context
    
    Returns:
        dict: Racks list result
    """
    import sys
    print(f"DEBUG: Starting racks list query, site: {site_name}, format: {format_type}", file=sys.stderr, flush=True)
    
    try:
        server_params = get_server_params()
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                tool_arguments = {}
                if site_name:
                    tool_arguments["site_name"] = site_name
                if format_type:
                    tool_arguments["format"] = format_type
                if conversation_history:
                    tool_arguments["conversation_history"] = conversation_history
                
                tool_result = await session.call_tool(
                    "list_racks",
                    arguments=tool_arguments
                )
                
                if tool_result and tool_result.content:
                    import json
                    result_text = tool_result.content[0].text
                    print(f"DEBUG: Racks list result text length: {len(result_text)}", file=sys.stderr, flush=True)
                    try:
                        result = json.loads(result_text)
                        print(f"DEBUG: Racks list result parsed successfully", file=sys.stderr, flush=True)
                        return result
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: JSON decode error: {e}", file=sys.stderr, flush=True)
                        return {"result": result_text}
                else:
                    print(f"DEBUG: No result content returned", file=sys.stderr, flush=True)
                    return None
    except asyncio.TimeoutError:
        return {"error": "Racks list lookup timed out"}
    except Exception as e:
        import sys
        print(f"DEBUG: Error executing racks list query: {str(e)}", file=sys.stderr, flush=True)
        return {"error": f"Error executing query: {str(e)}"}


async def execute_panorama_address_group_members_query(address_group_name, device_group=None, vsys="vsys1"):
    """
    Execute Panorama address group members query via MCP server.
    
    Args:
        address_group_name: Address group name to query
        device_group: Optional device group name
        vsys: VSYS name (default: "vsys1")
    
    Returns:
        dict: Address group members information
    """
    import sys
    print(f"DEBUG: Starting Panorama address group members query for group: {address_group_name}, device_group: {device_group}", file=sys.stderr, flush=True)
    
    try:
        print(f"DEBUG: Connecting to MCP server for Panorama address group members query...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized for Panorama address group members query", file=sys.stderr, flush=True)
            
            tool_arguments = {"address_group_name": address_group_name}
            if device_group:
                tool_arguments["device_group"] = device_group
            if vsys:
                tool_arguments["vsys"] = vsys
            
            print(f"DEBUG: Calling query_panorama_address_group_members with arguments: {tool_arguments}", file=sys.stderr, flush=True)
            
            # Detect client type for fallback
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
            
            print(f"DEBUG: Client type for Panorama query: FastMCP={is_fastmcp}", file=sys.stderr, flush=True)
            
            # Try standard format first (safest), then FastMCP format if needed
            try:
                # Standard MCP ClientSession format: pass arguments as a dictionary
                print(f"DEBUG: Trying standard format (arguments=dict) for query_panorama_address_group_members...", file=sys.stderr, flush=True)
                tool_result = await asyncio.wait_for(
                    client_or_session.call_tool("query_panorama_address_group_members", arguments=tool_arguments),
                    timeout=60.0
                )
                print(f"DEBUG: Standard format succeeded for query_panorama_address_group_members", file=sys.stderr, flush=True)
            except TypeError as e:
                error_str = str(e)
                # If standard format fails with "unexpected keyword argument 'arguments'", try FastMCP format
                if "unexpected keyword argument 'arguments'" in error_str and is_fastmcp:
                    print(f"DEBUG: Standard format failed, trying FastMCP format (keyword args) for query_panorama_address_group_members...", file=sys.stderr, flush=True)
                    tool_result_list = await asyncio.wait_for(
                        client_or_session.call_tool("query_panorama_address_group_members", **tool_arguments),
                        timeout=60.0
                    )
                    # Convert FastMCP result to standard format
                    if tool_result_list and len(tool_result_list) > 0:
                        class FastMCPToolResult:
                            def __init__(self, results):
                                self.content = []
                                for r in results:
                                    text = r.text if hasattr(r, 'text') else str(r)
                                    self.content.append(type('Content', (), {'text': text})())
                        tool_result = FastMCPToolResult(tool_result_list)
                        print(f"DEBUG: FastMCP format succeeded for query_panorama_address_group_members", file=sys.stderr, flush=True)
                    else:
                        tool_result = None
                else:
                    # Re-raise if it's a different error
                    print(f"DEBUG: Error calling query_panorama_address_group_members: {e}", file=sys.stderr, flush=True)
                    raise
            
            if tool_result and tool_result.content:
                result_text = tool_result.content[0].text
                print(f"DEBUG: Panorama address group members result text length: {len(result_text)}", file=sys.stderr, flush=True)
                try:
                    result = json.loads(result_text)
                    print(f"DEBUG: Panorama address group members result parsed successfully", file=sys.stderr, flush=True)
                    return result
                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON decode error: {e}", file=sys.stderr, flush=True)
                    return {"result": result_text}
            return None
    except asyncio.TimeoutError:
        print(f"DEBUG: Panorama address group members query timed out", file=sys.stderr, flush=True)
        return {"error": "Panorama query timed out. Please try again."}
    except Exception as e:
        print(f"DEBUG: Error in Panorama address group members query: {str(e)}", file=sys.stderr, flush=True)
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": f"Error querying Panorama: {str(e)}"}


async def execute_panorama_ip_object_group_query(ip_address, device_group=None, vsys="vsys1"):
    """
    Execute Panorama IP object group query via MCP server.
    
    Args:
        ip_address: IP address to search for
        device_group: Optional device group name
        vsys: VSYS name (default: "vsys1")
    
    Returns:
        dict: Object group information
    """
    import sys
    print(f"DEBUG: Starting Panorama IP object group query for IP: {ip_address}, device_group: {device_group}", file=sys.stderr, flush=True)
    
    try:
        print(f"DEBUG: Connecting to MCP server for Panorama IP object group query...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized for Panorama IP object group query", file=sys.stderr, flush=True)
            
            tool_arguments = {"ip_address": ip_address}
            if device_group:
                tool_arguments["device_group"] = device_group
            if vsys:
                tool_arguments["vsys"] = vsys
            
            print(f"DEBUG: Calling query_panorama_ip_object_group with arguments: {tool_arguments}", file=sys.stderr, flush=True)
            
            # Detect client type for fallback
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
            
            print(f"DEBUG: Client type for Panorama query: FastMCP={is_fastmcp}", file=sys.stderr, flush=True)
            
            # Try standard format first (safest), then FastMCP format if needed
            try:
                # Standard MCP ClientSession format: pass arguments as a dictionary
                print(f"DEBUG: Trying standard format (arguments=dict) for query_panorama_ip_object_group...", file=sys.stderr, flush=True)
                tool_result = await asyncio.wait_for(
                    client_or_session.call_tool("query_panorama_ip_object_group", arguments=tool_arguments),
                    timeout=60.0
                )
                print(f"DEBUG: Standard format succeeded for query_panorama_ip_object_group", file=sys.stderr, flush=True)
            except TypeError as e:
                error_str = str(e)
                # If standard format fails with "unexpected keyword argument 'arguments'", try FastMCP format
                if "unexpected keyword argument 'arguments'" in error_str and is_fastmcp:
                    print(f"DEBUG: Standard format failed, trying FastMCP format (keyword args) for query_panorama_ip_object_group...", file=sys.stderr, flush=True)
                    tool_result_list = await asyncio.wait_for(
                        client_or_session.call_tool("query_panorama_ip_object_group", **tool_arguments),
                        timeout=60.0
                    )
                    # Convert FastMCP result to standard format
                    if tool_result_list and len(tool_result_list) > 0:
                        class FastMCPToolResult:
                            def __init__(self, results):
                                self.content = []
                                for r in results:
                                    text = r.text if hasattr(r, 'text') else str(r)
                                    self.content.append(type('Content', (), {'text': text})())
                        tool_result = FastMCPToolResult(tool_result_list)
                        print(f"DEBUG: FastMCP format succeeded for query_panorama_ip_object_group", file=sys.stderr, flush=True)
                    else:
                        tool_result = None
                else:
                    # Re-raise if it's a different error
                    print(f"DEBUG: Error calling query_panorama_ip_object_group: {e}", file=sys.stderr, flush=True)
                    raise
            
            if tool_result and tool_result.content:
                result_text = tool_result.content[0].text
                print(f"DEBUG: Panorama result text length: {len(result_text)}", file=sys.stderr, flush=True)
                try:
                    result = json.loads(result_text)
                    print(f"DEBUG: Panorama result parsed successfully", file=sys.stderr, flush=True)
                    return result
                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON decode error: {e}", file=sys.stderr, flush=True)
                    return {"result": result_text}
            return None
    except asyncio.TimeoutError:
        return {"error": "Panorama query timed out"}
    except Exception as e:
        import sys
        import traceback
        print(f"DEBUG: Error executing Panorama query: {str(e)}", file=sys.stderr, flush=True)
        print(f"DEBUG: Full traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": f"Error executing query: {str(e)}"}


async def execute_splunk_recent_denies_query(ip_address, limit=100, earliest_time="-24h"):
    """
    Execute Splunk recent denies query via MCP server.

    Args:
        ip_address: IP address to search for in deny events
        limit: Max number of events (default 100)
        earliest_time: Splunk time range (default "-24h")

    Returns:
        dict: ip_address, events (list), count, and optional error
    """
    import sys
    print(f"DEBUG: Starting Splunk recent denies query for IP: {ip_address}", file=sys.stderr, flush=True)
    try:
        async for client_or_session in get_mcp_session():
            tool_arguments = {"ip_address": ip_address, "limit": limit, "earliest_time": earliest_time}
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
            try:
                tool_result = await asyncio.wait_for(
                    client_or_session.call_tool("get_splunk_recent_denies", arguments=tool_arguments),
                    timeout=90.0
                )
            except TypeError as e:
                if "unexpected keyword argument 'arguments'" in str(e) and is_fastmcp:
                    tool_result_list = await asyncio.wait_for(
                        client_or_session.call_tool("get_splunk_recent_denies", **tool_arguments),
                        timeout=90.0
                    )
                    if tool_result_list and len(tool_result_list) > 0:
                        class FastMCPToolResult:
                            def __init__(self, results):
                                self.content = []
                                for r in results:
                                    text = r.text if hasattr(r, 'text') else str(r)
                                    self.content.append(type('Content', (), {'text': text})())
                        tool_result = FastMCPToolResult(tool_result_list)
                    else:
                        tool_result = None
                else:
                    raise
            if tool_result and tool_result.content:
                result_text = tool_result.content[0].text
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    return {"result": result_text}
            return None
    except asyncio.TimeoutError:
        return {"error": "Splunk query timed out"}
    except Exception as e:
        import traceback
        print(f"DEBUG: Error executing Splunk query: {str(e)}", file=sys.stderr, flush=True)
        print(f"DEBUG: Full traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": f"Error executing query: {str(e)}"}


async def execute_rack_location_query(device_name, format_type=None, conversation_history=None, intent=None):
    """
    Execute rack location lookup via MCP server.

    Args:
        device_name: Device name to look up in NetBox
        format_type: Optional format request ("table", "json", "list")
        conversation_history: Optional conversation history for context
        intent: Optional intent ("device_details", "rack_location_only", "device_type_only", etc.)

    Returns:
        dict: Rack location result
    """
    import sys
    print(f"DEBUG: Starting rack location query for device: {device_name}, format: {format_type}, intent: {intent}", file=sys.stderr, flush=True)

    try:
        print(f"DEBUG: Connecting to MCP server for rack location query...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized for rack location query", file=sys.stderr, flush=True)
            
            tool_arguments = {"device_name": device_name}
            if format_type:
                tool_arguments["format"] = format_type
            if intent:
                tool_arguments["intent"] = intent
            if conversation_history:
                tool_arguments["conversation_history"] = conversation_history
            
            print(f"DEBUG: Calling get_device_rack_location with arguments: {tool_arguments}", file=sys.stderr, flush=True)
            
            # Detect client type and call appropriately
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
            
            print(f"DEBUG: Client type for rack location: FastMCP={is_fastmcp}, type: {type(client_or_session).__name__}, module: {type(client_or_session).__module__}", file=sys.stderr, flush=True)
            
            # Try standard format first (safest), then FastMCP format if needed
            try:
                # Standard MCP ClientSession format: pass arguments as a dictionary
                print(f"DEBUG: Trying standard format (arguments=dict) for get_device_rack_location...", file=sys.stderr, flush=True)
                tool_result = await asyncio.wait_for(
                    client_or_session.call_tool("get_device_rack_location", arguments=tool_arguments),
                    timeout=60.0
                )
                print(f"DEBUG: Standard format succeeded for get_device_rack_location", file=sys.stderr, flush=True)
            except TypeError as e:
                error_str = str(e)
                # If standard format fails with "unexpected keyword argument 'arguments'", try FastMCP format
                if "unexpected keyword argument 'arguments'" in error_str and is_fastmcp:
                    print(f"DEBUG: Standard format failed, trying FastMCP format (keyword args) for get_device_rack_location...", file=sys.stderr, flush=True)
                    tool_result_list = await asyncio.wait_for(
                        client_or_session.call_tool("get_device_rack_location", **tool_arguments),
                        timeout=60.0
                    )
                    # Convert FastMCP result to standard format
                    if tool_result_list and len(tool_result_list) > 0:
                        class FastMCPToolResult:
                            def __init__(self, results):
                                self.content = []
                                for r in results:
                                    text = r.text if hasattr(r, 'text') else str(r)
                                    self.content.append(type('Content', (), {'text': text})())
                        tool_result = FastMCPToolResult(tool_result_list)
                        print(f"DEBUG: FastMCP format succeeded for get_device_rack_location", file=sys.stderr, flush=True)
                    else:
                        tool_result = None
                else:
                    # Re-raise if it's a different error
                    print(f"DEBUG: Error calling get_device_rack_location: {e}", file=sys.stderr, flush=True)
                    raise
            
            if tool_result and tool_result.content:
                result_text = tool_result.content[0].text
                print(f"DEBUG: Raw result text length: {len(result_text)}", file=sys.stderr, flush=True)
                print(f"DEBUG: Raw result text preview: {result_text[:500]}", file=sys.stderr, flush=True)
                try:
                    result = json.loads(result_text)
                    print(f"DEBUG: Rack location result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}", file=sys.stderr, flush=True)
                    if isinstance(result, dict) and "ai_analysis" in result:
                        print(f"DEBUG: AI analysis found in result!", file=sys.stderr, flush=True)
                        print(f"DEBUG: AI analysis type: {type(result['ai_analysis'])}, content: {str(result['ai_analysis'])[:500]}", file=sys.stderr, flush=True)
                    else:
                        print(f"DEBUG: No AI analysis in result. Available keys: {list(result.keys())}", file=sys.stderr, flush=True)
                    return result
                except json.JSONDecodeError:
                    return {"result": result_text}
            return None
    except asyncio.TimeoutError:
        return {"error": "Rack location lookup timed out. Please try again."}
    except Exception as e:
        import traceback
        print(f"DEBUG: Error in execute_rack_location_query: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": f"Error executing rack location lookup: {str(e)}"}

def main():
    """
    Main function that creates and manages the chatbot interface.
    
    This function:
    1. Creates a chat interface for network path queries
    2. Maintains conversation history
    3. Parses natural language queries
    4. Displays results in chat format
    """
    print(f"DEBUG: main() function called", file=sys.stderr, flush=True)
    # Display the main page title
    st.title("🌐 Network Management Assistant")
    st.markdown("Ask me about network paths, device locations, or Panorama address groups! Try: *'Find path from 10.0.0.1 to 10.0.1.1 using TCP port 80'* or *'What address group is 11.0.0.0/24 part of?'*")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        default_live_data = st.checkbox(
            "Default: Use Live Data",
            value=True,
            help="Default setting for live data access. Can be overridden in queries."
        )
        st.session_state['default_live_data'] = default_live_data
        
        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        st.markdown("""
        **Splunk:**
        - *List all the denies for 10.0.0.250*
        - *Get recent deny events for 192.168.1.1*
        - *Show deny events for IP 10.0.1.100*
        
        **Panorama:**
        - *What address group is 11.0.0.0/24 part of?*
        - *Which address group contains 11.0.0.1?*
        - *What IPs are in the address group leander_web?*
        - *List all IPs in address group web_servers*
        
        **NetBox:**
        - *Where is leander-dc-border-leaf1 racked?*
        - *Show me rack details for A4*
        - *What is the rack location of leander-dc-leaf6?*
        
        **NetBrain:**
        - *Find path from 10.0.0.1 to 10.0.1.1*
        - *Query path from 192.168.1.10 to 192.168.2.20 using TCP port 443*
        - *Is traffic from 10.0.0.1 to 10.0.1.1 on TCP port 80 allowed?*
        - *Check if path from 10.0.0.250 to 10.0.1.250 is allowed on TCP 443*
        """)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your Network Management Assistant. I can help you with:\n\n• **NetBox Queries**: Find device rack locations and rack details\n• **Panorama Queries**: Query address groups, find which groups contain an IP, or list IPs in a group\n• **Network Path Queries**: Find network paths between devices\n• **Port / Path Allowed Check**: Check if traffic between two IPs on a given protocol/port is allowed or denied by policy\n\nJust ask me a question! For example: *'What address group is 11.0.0.0/24 part of?'* or *'Is traffic from 10.0.0.1 to 10.0.1.1 on TCP port 80 allowed?'*"
            }
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        try:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    # Assistant messages can contain various content types
                    if isinstance(message["content"], dict):
                        # Check if this is a racks list result FIRST (has 'racks' key with list value)
                        if "racks" in message["content"] and isinstance(message["content"].get("racks"), list):
                            # Display racks list result
                            format_type = message["content"].get("format_output")
                            display_racks_list_result(message["content"], st.container(), format_type)
                        # Check if it's a rack details result (has 'rack' key and 'devices_count' or 'devices')
                        elif "rack" in message["content"] and ("devices_count" in message["content"] or "devices" in message["content"]):
                            # Display rack details result
                            format_type = message["content"].get("format_output")
                            display_rack_details_result(message["content"], st.container(), format_type)
                        # Check if it's a rack location result (has 'rack' or 'device' key)
                        elif "rack" in message["content"] or "device" in message["content"]:
                            # Display rack location result (device query)
                            format_type = message["content"].get("format_output")  # Get format from result if stored
                            intent = message["content"].get("intent_output")  # Get intent from result if stored
                            # Restore yes/no question state if present (for proper re-rendering)
                            yes_no_question = message["content"].get("yes_no_question")
                            display_rack_location_result(message["content"], st.container(), format_type, intent=intent, yes_no_question=yes_no_question)
                        elif isinstance(message["content"], dict) and "ip_address" in message["content"] and ("address_objects" in message["content"] or "address_groups" in message["content"] or "device_group" in message["content"] or "vsys" in message["content"] or "error" in message["content"]):
                            # Display Panorama IP object group result
                            result = message["content"]
                            if isinstance(result, dict) and "error" in result:
                                st.error(f"❌ {result['error']}")
                            elif isinstance(result, dict) and "ai_analysis" in result:
                                # Display LLM summary (should be in table format)
                                ai_analysis = result["ai_analysis"]
                                if isinstance(ai_analysis, dict):
                                    summary = ai_analysis.get("summary")
                                    if summary:
                                        st.markdown(summary)
                                elif isinstance(ai_analysis, str):
                                    st.markdown(ai_analysis)
                            else:
                                st.info("Query completed. No results found or analysis unavailable.")
                        elif isinstance(message["content"], dict) and "status" in message["content"] and message["content"].get("status") in ("allowed", "denied", "unknown") and ("source" in message["content"] or "destination" in message["content"]):
                            # Display check_path_allowed result (has status: allowed/denied/unknown)
                            display_path_allowed_result(message["content"], st.container())
                        elif isinstance(message["content"], dict) and ("path_hops" in message["content"] or "simplified_hops" in message["content"] or "path_details" in message["content"]):
                            # Display network path result
                            display_result_chat(message["content"], st.container())
                        elif isinstance(message["content"], dict) and "events" in message["content"] and "ip_address" in message["content"]:
                            # Splunk recent denies result
                            result = message["content"]
                            if result.get("error"):
                                st.error(f"❌ {result['error']}")
                            else:
                                events = result.get("events", [])
                                count = result.get("count", len(events))
                                ip_address = result.get("ip_address", "")
                                st.success(f"Found **{count}** recent deny event(s) for **{ip_address}** from Splunk.")
                                if events:
                                    import pandas as pd
                                    df = pd.DataFrame(events)
                                    st.dataframe(df, width="stretch")
                                else:
                                    st.info("No deny events found for this IP in the time range.")
                        elif isinstance(message["content"], dict):
                            # Other dict results - try display_result_chat but it should handle unknown types gracefully
                            display_result_chat(message["content"], st.container())
                    else:
                        st.markdown(message["content"])
        except Exception as e:
            # If there's an error displaying a message, show an error but don't break the loop
            # This ensures all messages are still displayed even if one fails
            print(f"DEBUG: Error displaying message: {e}", file=sys.stderr, flush=True)
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
            with st.chat_message("assistant"):
                st.error(f"Error displaying message: {str(e)}")
                st.code(str(message.get("content", "Unknown content"))[:200])
    
    # Display buttons for all pending queries (so they're always visible and clickable)
    for key in list(st.session_state.keys()):
        if key.startswith('pending_query_') and isinstance(st.session_state[key], dict):
            query_data = st.session_state[key]
            # Ensure query_data is a dict
            if not isinstance(query_data, dict):
                print(f"DEBUG: Warning: {key} is not a dict: {type(query_data)}", file=sys.stderr, flush=True)
                continue
            if not query_data.get('confirmed', False):
                query_id = key.split('_')[-1]
                # Display buttons for this pending query
                with st.chat_message("assistant"):
                    st.info(f"📋 Path query from **{query_data.get('source', 'Unknown')}** to **{query_data.get('destination', 'Unknown')}** using **{query_data.get('protocol', 'TCP')}** port **{query_data.get('port', '0')}**.")
                    st.markdown("**Please choose the data source:**")
                    col1, col2 = st.columns(2)
                    
                    suggested_live = query_data.get('suggested_live', True) if isinstance(query_data, dict) else True
                    with col1:
                        use_live = st.button(
                            "🔴 Use Live Data",
                            key=f"live_btn_{query_id}",
                            width="stretch",
                            type="primary" if suggested_live else "secondary",
                            help="Use real-time live access data (may take longer but more current)"
                        )
                    
                    with col2:
                        use_baseline = st.button(
                            "💾 Use Cached/Baseline Data",
                            key=f"baseline_btn_{query_id}",
                            width="stretch",
                            type="primary" if not suggested_live else "secondary",
                            help="Use cached baseline data (faster but may be older)"
                        )
                    
                    # If user clicked a button, store the choice and trigger execution
                    button_click_key = f"button_clicked_{query_id}"
                    if use_live:
                        print(f"DEBUG: Live button clicked! key={key}", file=sys.stderr, flush=True)
                        st.session_state[button_click_key] = "live"
                        st.session_state[key]['is_live'] = True
                        st.session_state[key]['confirmed'] = True
                        st.rerun()
                    elif use_baseline:
                        print(f"DEBUG: Baseline button clicked! key={key}", file=sys.stderr, flush=True)
                        st.session_state[button_click_key] = "baseline"
                        st.session_state[key]['is_live'] = False
                        st.session_state[key]['confirmed'] = True
                        st.rerun()
    
    # Check for button clicks on ALL pending queries (runs on every rerun, BEFORE chat input)
    # Also check button states directly since buttons reset after rerun
    for key in list(st.session_state.keys()):
        if key.startswith('pending_query_') and isinstance(st.session_state[key], dict):
            query_id = key.split('_')[-1]
            button_click_key = f"button_clicked_{query_id}"
            
            # Check if button was clicked (check button state directly)
            live_btn_key = f"live_btn_{query_id}"
            baseline_btn_key = f"baseline_btn_{query_id}"
            
            # Check if buttons exist in widget state (Streamlit's internal state)
            if live_btn_key in st.session_state:
                if st.session_state[live_btn_key]:
                    print(f"DEBUG: Live button state is True for {key}", file=sys.stderr, flush=True)
                    st.session_state[button_click_key] = "live"
                    st.session_state[key]['is_live'] = True
                    st.session_state[key]['confirmed'] = True
                    print(f"DEBUG: Setting confirmed=True for {key}", file=sys.stderr, flush=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Query confirmed. Using live data..."
                    })
                    st.rerun()
            elif baseline_btn_key in st.session_state:
                if st.session_state[baseline_btn_key]:
                    print(f"DEBUG: Baseline button state is True for {key}", file=sys.stderr, flush=True)
                    st.session_state[button_click_key] = "baseline"
                    st.session_state[key]['is_live'] = False
                    st.session_state[key]['confirmed'] = True
                    print(f"DEBUG: Setting confirmed=True for {key}", file=sys.stderr, flush=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Query confirmed. Using baseline data..."
                    })
                    st.rerun()
            elif button_click_key in st.session_state:
                # Button was clicked, set confirmed
                data_type = st.session_state[button_click_key]
                is_live_choice = (data_type == "live")
                st.session_state[key]['is_live'] = is_live_choice
                st.session_state[key]['confirmed'] = True
                print(f"DEBUG: Button click detected for {key}: {data_type}, setting confirmed=True", file=sys.stderr, flush=True)
                # Clear the button click tracker
                del st.session_state[button_click_key]
                # Add confirmation message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Query confirmed. Using {data_type} data..."
                })
                st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask about a network path..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get conversation history for context-aware responses
        conversation_history = [
            {"role": msg.get("role", ""), "content": str(msg.get("content", ""))}
            for msg in st.session_state.messages[-20:-1]  # Exclude current message, last 20 messages
            if isinstance(msg.get("content"), str)  # Only include text messages for context
        ]
        
        # Check if this looks like just a site name (for context inference from previous queries)
        # Pattern: just location words, no rack keywords, no device names, no numbers
        prompt_lower = prompt.lower().strip()
        is_likely_site_name = (
            len(prompt.split()) <= 3 and  # Short query (1-3 words)
            not any(char.isdigit() for char in prompt) and  # No numbers (rack names have numbers)
            "-" not in prompt and  # No dashes (device names have dashes)
            "rack" not in prompt_lower and  # No "rack" keyword
            "device" not in prompt_lower and  # No "device" keyword
            "path" not in prompt_lower and  # No "path" keyword
            "in" not in prompt_lower  # No "in" keyword (would be "A1 in round rock")
        )
        
        # PRIORITY: Check if this is a follow-up response to a site clarification question FIRST
        # This is the active pending query that needs site information
        last_rack_query = st.session_state.get("last_rack_query")
        if last_rack_query and is_likely_site_name:
            # User is providing site name for previous rack query
            rack_name = last_rack_query.get("rack_name")
            site_name = prompt.strip()
            format_type = last_rack_query.get("format_type", "table")
            
            print(f"DEBUG: Using last_rack_query with rack_name={rack_name}, site_name={site_name}", file=sys.stderr, flush=True)
            
            # Clear the stored query IMMEDIATELY to prevent reuse
            st.session_state["last_rack_query"] = None
            
            # Execute the rack query with the site name
            status_msg = f"🔎 Looking up rack details for **{rack_name}** in **{site_name}**..."
            with st.chat_message("assistant"):
                st.info(status_msg)
            
            try:
                max_timeout = 60
                try:
                    asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(
                                asyncio.wait_for(
                                    execute_rack_details_query(rack_name, format_type, conversation_history, site_name=site_name),
                                    timeout=max_timeout
                                )
                            )
                        )
                        result = future.result(timeout=max_timeout + 5)
                except RuntimeError:
                    result = asyncio.run(
                        asyncio.wait_for(
                            execute_rack_details_query(rack_name, format_type, conversation_history, site_name=site_name),
                            timeout=max_timeout
                        )
                    )
                
                with st.chat_message("assistant"):
                    display_rack_details_result(result, st.container(), format_type)
                
                # Store result with format_type for proper re-rendering
                result_with_format = result.copy() if isinstance(result, dict) else result
                if isinstance(result_with_format, dict):
                    result_with_format["format_output"] = format_type
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result_with_format
                })
                
                # Store rack context for future queries (if this was a rack query)
                if rack_name:
                    st.session_state["last_rack_context"] = {
                        "rack_name": rack_name,
                        "site_name": site_name,
                        "format_type": format_type or "table"
                    }
                
                return  # Exit after successful lookup
            except asyncio.TimeoutError:
                error_msg = "⏱️ Rack details lookup timed out. Please try again."
                with st.chat_message("assistant"):
                    st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                with st.chat_message("assistant"):
                    st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return
        
        # Simple safety check: Detect obvious rack query patterns before LLM
        # This prevents LLM from misclassifying obvious rack queries as device queries
        prompt_lower = prompt.lower().strip()
        has_rack_keyword = any(phrase in prompt_lower for phrase in [
            "rack details", "rack info", "show rack", "what's in rack", 
            "devices in rack", "rack details for", "rack info for"
        ])
        
        # Use tool discovery to select the appropriate tool based on tool descriptions
        # NO regex pattern matching - let the LLM decide based on tool descriptions
        # NO pattern matching, NO pre-checks - let the LLM use tool descriptions to decide
        async def discover_and_execute_tool():
            try:
                print(f"DEBUG: Creating MCP server connection...", file=sys.stderr, flush=True)
                # Use the same HTTP connection approach as execute_network_query
                async for client_or_session in get_mcp_session():
                    print(f"DEBUG: Session created, initializing...", file=sys.stderr, flush=True)
                    try:
                        print(f"DEBUG: Session initialized successfully", file=sys.stderr, flush=True)
                        
                        # Get available tools and their descriptions
                        # FastMCP Client and standard MCP ClientSession have different APIs
                        if FASTMCP_CLIENT_AVAILABLE and isinstance(client_or_session, FastMCPClient):
                            # FastMCP Client - check available methods
                            if hasattr(client_or_session, 'list_tools'):
                                tools_result = await client_or_session.list_tools()
                                # FastMCP might return a list directly
                                tools = tools_result if isinstance(tools_result, list) else (tools_result.tools if hasattr(tools_result, 'tools') else [])
                            else:
                                # Try alternative method or get tools from server info
                                print(f"DEBUG: FastMCP Client doesn't have list_tools, trying alternative...", file=sys.stderr, flush=True)
                                tools = []
                        else:
                            # Standard MCP ClientSession
                            tools_result = await client_or_session.list_tools()
                            if isinstance(tools_result, list):
                                tools = tools_result
                            elif hasattr(tools_result, 'tools'):
                                tools = tools_result.tools if tools_result else []
                            else:
                                tools = tools_result if tools_result else []
                        
                        print(f"DEBUG: Found {len(tools)} available tools", file=sys.stderr, flush=True)
                        
                        # Build tool descriptions for LLM - format with clear structure
                        tool_names_list = [tool.name for tool in tools]
                        tools_description = f"\n\n**AVAILABLE TOOL NAMES (use EXACTLY as shown): {', '.join(tool_names_list)}**\n\n" + "="*80 + "\n\n".join([
                            f"Tool Name: {tool.name}\n\nDescription:\n{tool.description or 'No description'}\n\nParameters: {', '.join([p for p in (tool.inputSchema.get('properties', {}).keys() if isinstance(tool.inputSchema, dict) else [])])}"
                            for tool in tools
                        ]) + "\n\n" + "="*80
                        
                        print(f"DEBUG: Tools description: {tools_description[:500]}...", file=sys.stderr, flush=True)
                        
                        # Use production-grade tool selection module
                        from mcp_client_tool_selection import select_tool_with_llm
                        
                        print(f"DEBUG: Calling select_tool_with_llm for prompt: '{prompt}'", file=sys.stderr, flush=True)
                        tool_selection_result = await select_tool_with_llm(
                            prompt=prompt,
                            tools_description=tools_description,
                            conversation_history=conversation_history
                        )
                        
                        print(f"DEBUG: select_tool_with_llm returned: success={tool_selection_result.get('success')}, needs_clarification={tool_selection_result.get('needs_clarification')}, tool_name={tool_selection_result.get('tool_name')}, error={tool_selection_result.get('error')}", file=sys.stderr, flush=True)
                        
                        if not tool_selection_result.get("success"):
                            print(f"DEBUG: Tool selection failed, returning error result", file=sys.stderr, flush=True)
                            return tool_selection_result
                        
                        # Check if tool_name is None and it's not a clarification request
                        # This means the query cannot be processed by any available tool
                        tool_name = tool_selection_result.get("tool_name")
                        needs_clarification = tool_selection_result.get("needs_clarification", False)
                        if tool_name is None and not needs_clarification:
                            # Query cannot be processed - return appropriate message
                            clarification_msg = tool_selection_result.get("clarification_question")
                            if not clarification_msg:
                                clarification_msg = "I'm sorry, but this system is not equipped to process that type of query. I can only help with:\n- Network path queries\n- Device rack location lookups\n- Rack details queries\n- Panorama address group queries\n\nCould you please rephrase your query to match one of these capabilities?"
                            return {
                                "success": False,
                                "tool_name": None,
                                "needs_clarification": False,
                                "clarification_question": clarification_msg,
                                "error": "Query cannot be processed by any available tool"
                            }
                        # Check if clarification is needed
                        if tool_selection_result.get("needs_clarification", False):
                            clarification_question = tool_selection_result.get("clarification_question", "Could you please clarify what you're looking for?")
                            print(f"DEBUG: LLM requested clarification: {clarification_question}", file=sys.stderr, flush=True)
                            return {
                                "success": False,
                                "needs_clarification": True,
                                "clarification_question": clarification_question
                            }
                        
                        selected_tool = tool_selection_result.get("tool_name")
                        tool_params = tool_selection_result.get("parameters", {})
                        format_type = tool_selection_result.get("format", "table")
                        intent = tool_selection_result.get("intent")
                        
                        # Move intent from parameters to top level if needed (don't pass it to tool)
                        if "intent" in tool_params:
                            intent = tool_params.pop("intent")
                        
                        print(f"DEBUG: Selected tool: {selected_tool}, params: {tool_params}, format: {format_type}, intent: {intent}", file=sys.stderr, flush=True)
                        
                        if not selected_tool:
                            print(f"DEBUG: ERROR - LLM did not return a tool_name", file=sys.stderr, flush=True)
                            # Check if there's a clarification message explaining why it can't be processed
                            clarification_msg = tool_selection_result.get("clarification_question")
                            if clarification_msg:
                                return {
                                    "success": False,
                                    "tool_name": None,
                                    "needs_clarification": False,
                                    "clarification_question": clarification_msg,
                                    "error": "Query cannot be processed by any available tool"
                                }
                            return {"success": False, "error": "LLM did not select a tool. Please ensure tool descriptions are clear."}
                        
                        return {
                            "success": True,
                            "tool_name": selected_tool,
                            "parameters": tool_params,
                            "format": format_type,
                            "intent": intent
                        }
                    except Exception as session_error:
                        import traceback
                        error_type = type(session_error).__name__
                        error_msg = str(session_error)
                        print(f"DEBUG: Error in session operations: {error_type}: {error_msg}", file=sys.stderr, flush=True)
                        print(f"DEBUG: Session error traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                        
                        # Provide more helpful error message for connection issues
                        if "Connection closed" in error_msg or "McpError" in error_type:
                            return {"success": False, "error": f"MCP server connection failed. The server may not be running or may have crashed. Error: {error_msg}"}
                        # Handle JSON parsing errors (server sending malformed JSON)
                        if "JSON" in error_msg or "json" in error_msg or "EOF while parsing" in error_msg or "json_invalid" in error_msg:
                            return {"success": False, "error": f"MCP server sent invalid JSON response. The server may have encountered an internal error. Check mcp_server.log for details. Error: {error_msg}"}
                        # Handle validation errors
                        if "validation error" in error_msg.lower() or "JSONRPCMessage" in error_msg:
                            return {"success": False, "error": f"MCP protocol error: Server sent malformed response. The server may have crashed. Check mcp_server.log for details. Error: {error_msg}"}
                        return {"success": False, "error": f"Error during tool discovery: {error_msg}"}
                    except Exception as client_error:
                        import traceback
                        error_type = type(client_error).__name__
                        error_msg = str(client_error)
                        print(f"DEBUG: Error in client session: {error_type}: {error_msg}", file=sys.stderr, flush=True)
                        print(f"DEBUG: Client error traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                        
                        # Handle JSON/protocol errors at client level
                        if "JSON" in error_msg or "json" in error_msg or "EOF while parsing" in error_msg or "json_invalid" in error_msg or "validation error" in error_msg.lower() or "JSONRPCMessage" in error_msg:
                            return {"success": False, "error": f"MCP protocol error: Server sent malformed JSON response. The server may have crashed. Check mcp_server.log for details. Error: {error_msg}"}
                        return {"success": False, "error": f"Error establishing MCP connection: {error_msg}"}
            except Exception as e:
                import traceback
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"DEBUG: Tool discovery failed: {error_type}: {error_msg}", file=sys.stderr, flush=True)
                print(f"DEBUG: Full traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                
                # Handle JSON/protocol errors at outer level
                if "JSON" in error_msg or "json" in error_msg or "EOF while parsing" in error_msg or "json_invalid" in error_msg or "validation error" in error_msg.lower() or "JSONRPCMessage" in error_msg:
                    return {"success": False, "error": f"MCP protocol error: Server sent malformed JSON response. The server may have crashed. Check mcp_server.log for details. Error: {error_msg}"}
                return {"success": False, "error": f"Tool discovery error: {error_msg}"}
        
        # No pre-checks - rely entirely on LLM + tool descriptions
        
        # Show "In progress" for the entire flow (discovery + tool execution) so the user knows the app is working
        with st.spinner("⏳ **In progress** — processing your query..."):
            try:
                try:
                    asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(discover_and_execute_tool())
                        )
                        tool_selection = future.result(timeout=30)
                except RuntimeError:
                    tool_selection = asyncio.run(discover_and_execute_tool())
                
                print(f"DEBUG: Tool selection result: success={tool_selection.get('success')}, needs_clarification={tool_selection.get('needs_clarification')}, tool_name={tool_selection.get('tool_name')}, error={tool_selection.get('error')}", file=sys.stderr, flush=True)
                
                # Check if query cannot be processed (no tool available and not a clarification request)
                tool_name = tool_selection.get("tool_name")
                needs_clarification = tool_selection.get("needs_clarification", False)
                if tool_name is None and not needs_clarification:
                    # Query cannot be processed by any available tool
                    clarification_msg = tool_selection.get("clarification_question")
                    if not clarification_msg:
                        clarification_msg = "I'm sorry, but this system is not equipped to process that type of query. I can only help with:\n- Network path queries (e.g., 'Find path from 10.0.0.1 to 10.0.1.1')\n- Device rack location lookups (e.g., 'Where is leander-dc-leaf6 racked?')\n- Rack details queries (e.g., 'Show rack details for A4')\n- Panorama address group queries (e.g., 'What address group is 11.0.0.0/24 part of?')\n\nCould you please rephrase your query to match one of these capabilities?"
                    with st.chat_message("assistant"):
                        st.warning(clarification_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": clarification_msg
                    })
                    return
                
                # Check if LLM requested clarification
                if tool_selection.get("needs_clarification", False):
                    clarification_question = tool_selection.get("clarification_question", "Could you please clarify what you're looking for?")
                    with st.chat_message("assistant"):
                        st.info(clarification_question)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": clarification_question
                    })
                    return
                
                if tool_selection.get("success") and tool_selection.get("tool_name"):
                    selected_tool = tool_selection["tool_name"]
                    
                    # Safeguard: If tool_name is "needs_clarification" or None, check for clarification message
                    if selected_tool == "needs_clarification" or selected_tool is None:
                        # Check if there's a clarification message explaining why it can't be processed
                        clarification_msg = tool_selection.get("clarification_question")
                        if clarification_msg:
                            with st.chat_message("assistant"):
                                st.warning(clarification_msg)
                            st.session_state.messages.append({"role": "assistant", "content": clarification_msg})
                        else:
                            error_msg = "I'm sorry, but this system is not equipped to process that type of query. I can only help with network path queries, device rack location lookups, rack details, and Panorama address group queries."
                            with st.chat_message("assistant"):
                                st.warning(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        return
                    
                    tool_params = tool_selection.get("parameters", {})
                    format_type = tool_selection.get("format")
                    intent = tool_selection.get("intent")
                
                    print(f"DEBUG: Executing tool: {selected_tool} with params: {tool_params}", file=sys.stderr, flush=True)
                
                    # Execute the selected tool - trust LLM's parameter extraction
                    if selected_tool == "get_device_rack_location":
                        device_name = tool_params.get("device_name", "").strip()
                        if not device_name:
                            # No fallback - LLM should have asked for clarification if device name was unclear
                            error_msg = "Device name not found in query. Please specify a device name."
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                    
                        # Check if this is a yes/no question (e.g., "is Arista the manufacturer for...")
                        prompt_lower = prompt.lower()
                        is_yes_no_question = False
                        expected_value = None
                        question_field = None
                    
                        # Pattern: "is X the manufacturer for device"
                        import re
                        manufacturer_match = re.search(r'is\s+([^?\s]+(?:\s+[^?\s]+)*?)\s+the\s+manufacturer\s+for', prompt_lower)
                        if manufacturer_match:
                            is_yes_no_question = True
                            expected_value = manufacturer_match.group(1).strip()
                            question_field = "manufacturer"
                            intent = "manufacturer_only"  # We need manufacturer to compare
                            print(f"DEBUG: Detected yes/no manufacturer question. Expected: {expected_value}", file=sys.stderr, flush=True)
                    
                        # Pattern: "is X the status for device"
                        status_match = re.search(r'is\s+([^?\s]+(?:\s+[^?\s]+)*?)\s+the\s+status\s+for', prompt_lower)
                        if status_match:
                            is_yes_no_question = True
                            expected_value = status_match.group(1).strip()
                            question_field = "status"
                            intent = "status_only"
                            print(f"DEBUG: Detected yes/no status question. Expected: {expected_value}", file=sys.stderr, flush=True)
                    
                        # Pattern: "is X the device type for device"
                        device_type_match = re.search(r'is\s+([^?\s]+(?:\s+[^?\s]+)*?)\s+the\s+device\s+type\s+for', prompt_lower)
                        if device_type_match:
                            is_yes_no_question = True
                            expected_value = device_type_match.group(1).strip()
                            question_field = "device_type"
                            intent = "device_type_only"
                            print(f"DEBUG: Detected yes/no device type question. Expected: {expected_value}", file=sys.stderr, flush=True)
                    
                        # Default to table format
                        if not format_type:
                            format_type = "table"
                    
                        # Use LLM-provided intent, or default to device_details if not provided
                        if not intent:
                            intent = "device_details"
                            print(f"DEBUG: No intent from LLM, defaulting to device_details", file=sys.stderr, flush=True)
                        else:
                            print(f"DEBUG: Using LLM-provided intent: {intent}", file=sys.stderr, flush=True)
                    
                        # Store yes/no question info for later comparison
                        if is_yes_no_question:
                            st.session_state["yes_no_question"] = {
                                "expected_value": expected_value,
                                "question_field": question_field
                            }
                    
                        # Ensure format is "table" for specific field intents
                        if intent in ("site_only", "status_only", "device_type_only", "manufacturer_only", "rack_location_only"):
                            format_type = "table"
                            print(f"DEBUG: Format set to 'table' for intent: {intent}", file=sys.stderr, flush=True)
                    
                        if device_name:
                            status_msg = f"🔎 Looking up device details for **{device_name}**..."
                            with st.chat_message("assistant"):
                                st.info(status_msg)

                            try:
                                max_timeout = 60
                                try:
                                    asyncio.get_running_loop()
                                    import concurrent.futures
                                    with concurrent.futures.ThreadPoolExecutor() as executor:
                                        future = executor.submit(
                                            lambda: asyncio.run(
                                                asyncio.wait_for(
                                                    execute_rack_location_query(device_name, format_type, conversation_history, intent),
                                                    timeout=max_timeout
                                                )
                                            )
                                        )
                                        result = future.result(timeout=max_timeout + 5)
                                except RuntimeError:
                                    result = asyncio.run(
                                        asyncio.wait_for(
                                            execute_rack_location_query(device_name, format_type, conversation_history, intent),
                                            timeout=max_timeout
                                        )
                                    )

                                with st.chat_message("assistant"):
                                    print(f"DEBUG: About to display result with intent: {intent}, format: {format_type}", file=sys.stderr, flush=True)
                                    print(f"DEBUG: Result keys before display: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}", file=sys.stderr, flush=True)
                                    print(f"DEBUG: Result intent field: {result.get('intent') if isinstance(result, dict) else 'N/A'}", file=sys.stderr, flush=True)
                                    display_rack_location_result(result, st.container(), format_type, intent=intent)

                                # Store result with format_type and intent for proper re-rendering
                                result_with_format = result.copy() if isinstance(result, dict) else result
                                if isinstance(result_with_format, dict):
                                    result_with_format["format_output"] = format_type
                                    # Always store intent (even if None or device_details) so re-rendering works correctly
                                    result_with_format["intent_output"] = intent if intent else "device_details"
                                    # Store yes/no question state if present (for proper re-rendering)
                                    yes_no_question = st.session_state.get("yes_no_question")
                                    if yes_no_question:
                                        result_with_format["yes_no_question"] = yes_no_question
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": result_with_format
                                })
                                # Clear yes/no question state after storing (it's now in the message)
                                st.session_state.pop("yes_no_question", None)
                                return  # Exit after successful lookup
                            except asyncio.TimeoutError:
                                error_msg = "⏱️ Rack location lookup timed out. Please try again."
                                with st.chat_message("assistant"):
                                    st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                return
                            except Exception as e:
                                error_msg = f"An error occurred: {str(e)}"
                                with st.chat_message("assistant"):
                                    st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                return
                
                    elif selected_tool == "list_racks":
                        # Use LLM-extracted values
                        site_name = tool_params.get("site_name")
                        format_type = tool_params.get("format") or format_type or "table"
                    
                        status_msg = f"🔎 Looking up all racks"
                        if site_name:
                            status_msg = f"🔎 Looking up racks at **{site_name}**"
                        with st.chat_message("assistant"):
                            st.info(status_msg)
                    
                        try:
                            max_timeout = 60
                            try:
                                asyncio.get_running_loop()
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(
                                        lambda: asyncio.run(
                                            asyncio.wait_for(
                                                execute_racks_list_query(site_name, format_type, conversation_history),
                                                timeout=max_timeout
                                            )
                                        )
                                    )
                                    result = future.result(timeout=max_timeout + 5)
                            except RuntimeError:
                                result = asyncio.run(
                                    asyncio.wait_for(
                                        execute_racks_list_query(site_name, format_type, conversation_history),
                                        timeout=max_timeout
                                    )
                                )
                        
                            with st.chat_message("assistant"):
                                display_racks_list_result(result, st.container(), format_type)
                        
                            # Store result with format_type for proper re-rendering
                            result_with_format = result.copy() if isinstance(result, dict) else result
                            if isinstance(result_with_format, dict):
                                result_with_format["format_output"] = format_type
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result_with_format
                            })
                            return  # Exit after successful lookup
                        except asyncio.TimeoutError:
                            error_msg = "⏱️ Racks list lookup timed out. Please try again."
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                        except Exception as e:
                            error_msg = f"An error occurred: {str(e)}"
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                
                    elif selected_tool == "get_rack_details":
                        # Use LLM-extracted values - no regex needed
                        rack_name = tool_params.get("rack_name")
                        site_name = tool_params.get("site_name")
                    
                        # Default to table format if no format specified
                        if not format_type:
                            format_type = "table"
                    
                        if rack_name:
                            site_info = f" in **{site_name}**" if site_name else ""
                            status_msg = f"🔎 Looking up rack details for **{rack_name}**{site_info}..."
                            with st.chat_message("assistant"):
                                st.info(status_msg)
                        
                            # Clear any stored rack query since we're proceeding
                            if "last_rack_query" in st.session_state:
                                st.session_state["last_rack_query"] = None

                            try:
                                max_timeout = 60
                                try:
                                    asyncio.get_running_loop()
                                    import concurrent.futures
                                    with concurrent.futures.ThreadPoolExecutor() as executor:
                                        future = executor.submit(
                                            lambda: asyncio.run(
                                                asyncio.wait_for(
                                                    execute_rack_details_query(rack_name, format_type, conversation_history, site_name=site_name),
                                                    timeout=max_timeout
                                                )
                                            )
                                        )
                                        result = future.result(timeout=max_timeout + 5)
                                except RuntimeError:
                                    result = asyncio.run(
                                        asyncio.wait_for(
                                            execute_rack_details_query(rack_name, format_type, conversation_history, site_name=site_name),
                                            timeout=max_timeout
                                        )
                                    )

                                # Check if server returned an error or requires site clarification
                                if isinstance(result, dict) and "error" in result:
                                    error_msg = result.get("error", "")
                                    if "device name" in error_msg.lower() and "dash" in error_msg.lower():
                                        # Server correctly identified this as a device name - show error and let tool discovery handle it
                                        print(f"DEBUG: Server detected device name in rack query, error: {error_msg}", file=sys.stderr, flush=True)
                                        with st.chat_message("assistant"):
                                            st.error(f"❌ {error_msg}")
                                            if "suggestion" in result:
                                                st.info(result["suggestion"])
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": result
                                        })
                                        # Don't return - let it fall through to tool discovery which should use get_device_rack_location
                                    elif result.get("requires_site"):
                                        # Multiple racks with same name at different sites - ask for site clarification
                                        sites = result.get("sites", [])
                                        sites_list = ", ".join([f"'{s}'" for s in sites]) if sites else "different sites"
                                        clarifying_text = f"I found rack **{rack_name}** at multiple sites ({sites_list}). Please specify which site (e.g., 'Round Rock DC', 'Leander DC', etc.)."
                                    
                                        # Store the rack query for follow-up
                                        st.session_state["last_rack_query"] = {
                                            "rack_name": rack_name,
                                            "format_type": format_type or "table"
                                        }
                                        print(f"DEBUG: Set last_rack_query to rack_name={rack_name} (multiple sites found)", file=sys.stderr, flush=True)
                                    
                                        with st.chat_message("assistant"):
                                            st.info(clarifying_text)
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": clarifying_text
                                        })
                                        return  # Exit to wait for user's site response
                                    else:
                                        # Other error (e.g., "Rack not found") - display it
                                        with st.chat_message("assistant"):
                                            display_rack_details_result(result, st.container(), format_type)
                                        result_with_format = result.copy() if isinstance(result, dict) else result
                                        if isinstance(result_with_format, dict):
                                            result_with_format["format_output"] = format_type
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": result_with_format
                                        })
                                        return
                            
                                # Success - display result
                                with st.chat_message("assistant"):
                                    display_rack_details_result(result, st.container(), format_type)

                                    # Store result with format_type for proper re-rendering
                                    result_with_format = result.copy() if isinstance(result, dict) else result
                                    if isinstance(result_with_format, dict):
                                        result_with_format["format_output"] = format_type
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": result_with_format
                                    })
                                    return  # Exit after successful lookup
                            except asyncio.TimeoutError:
                                error_msg = "⏱️ Rack details lookup timed out. Please try again."
                                with st.chat_message("assistant"):
                                    st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                return
                            except Exception as e:
                                error_msg = f"An error occurred: {str(e)}"
                                with st.chat_message("assistant"):
                                    st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                return
                
                    elif selected_tool == "check_path_allowed":
                        source = tool_params.get("source", "").strip()
                        destination = tool_params.get("destination", "").strip()
                        protocol = tool_params.get("protocol", "TCP").strip()
                        port = tool_params.get("port", "0").strip()
                        is_live = True  # Default to live data
                    
                        if not source or not destination:
                            error_msg = "Source and destination IP addresses are required for path allowed check."
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                    
                        # Execute path allowed check
                        with st.chat_message("assistant"):
                            with st.spinner(f"Checking if traffic from {source} to {destination} on {protocol}/{port} is allowed..."):
                                try:
                                    try:
                                        asyncio.get_running_loop()
                                        import concurrent.futures
                                        with concurrent.futures.ThreadPoolExecutor() as executor:
                                            future = executor.submit(
                                                lambda: asyncio.run(execute_path_allowed_check(source, destination, protocol, port, is_live))
                                            )
                                            result = future.result(timeout=380)  # 380 seconds timeout
                                    except RuntimeError:
                                        result = asyncio.run(execute_path_allowed_check(source, destination, protocol, port, is_live))
                                
                                    # Display result
                                    display_path_allowed_result(result, st.container())
                                
                                    # Store result in conversation history
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": result
                                    })
                                except asyncio.TimeoutError:
                                    error_msg = "⏱️ Path allowed check timed out. Please try again."
                                    st.error(error_msg)
                                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                except Exception as e:
                                    error_msg = f"An error occurred: {str(e)}"
                                    st.error(error_msg)
                                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        return
                
                    elif selected_tool == "query_network_path":
                        # Handle network path query
                        source = tool_params.get("source")
                        destination = tool_params.get("destination")
                        protocol = tool_params.get("protocol", "TCP")
                        port = tool_params.get("port", "0")
                    
                        if source and destination:
                            # Execute network path query
                            default_live = st.session_state.get('default_live_data', True)
                            is_live = tool_params.get("is_live", 1 if default_live else 0)
                        
                            try:
                                # Network path queries can take a long time (server polls for task completion)
                                # Set timeout to 380 seconds to allow for server processing (tool call timeout is 360s)
                                max_timeout = 380
                                try:
                                    asyncio.get_running_loop()
                                    import concurrent.futures
                                    with concurrent.futures.ThreadPoolExecutor() as executor:
                                        future = executor.submit(
                                            lambda: asyncio.run(
                                                asyncio.wait_for(
                                                    execute_network_query(source, destination, protocol, port, is_live),
                                                    timeout=max_timeout
                                                )
                                            )
                                        )
                                        result = future.result(timeout=max_timeout + 10)  # Add buffer for thread overhead
                                except RuntimeError:
                                    result = asyncio.run(
                                        asyncio.wait_for(
                                            execute_network_query(source, destination, protocol, port, is_live),
                                            timeout=max_timeout
                                        )
                                    )
                            
                                # Unwrap result if it's wrapped in {"result": ...}
                                if isinstance(result, dict) and "result" in result and len(result) == 1:
                                    # Check if the inner result is a string that might be JSON
                                    inner_result = result["result"]
                                    if isinstance(inner_result, str):
                                        try:
                                            import json
                                            unwrapped = json.loads(inner_result)
                                            print(f"DEBUG: Unwrapped result from string, keys: {list(unwrapped.keys()) if isinstance(unwrapped, dict) else 'not a dict'}", file=sys.stderr, flush=True)
                                            result = unwrapped
                                        except (json.JSONDecodeError, TypeError):
                                            # If it's not JSON, keep the wrapped version
                                            pass
                            
                                with st.chat_message("assistant"):
                                    display_result_chat(result, st.container())
                            
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": result
                                })
                                return
                            except Exception as e:
                                error_msg = f"An error occurred: {str(e)}"
                                with st.chat_message("assistant"):
                                    st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                return
                
                    elif selected_tool == "query_panorama_ip_object_group":
                        # Handle Panorama IP object group query
                        ip_address = tool_params.get("ip_address", "").strip()
                        device_group = tool_params.get("device_group")
                        vsys = tool_params.get("vsys", "vsys1")
                    
                        if not ip_address:
                            error_msg = "IP address not found in query. Please specify an IP address (e.g., 'what address group 10.0.0.254 belongs to')."
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                    
                        status_msg = f"🔎 Querying Panorama for address groups containing **{ip_address}**..."
                        with st.chat_message("assistant"):
                            st.info(status_msg)
                    
                        try:
                            max_timeout = 60
                            try:
                                asyncio.get_running_loop()
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(
                                        lambda: asyncio.run(
                                            asyncio.wait_for(
                                                execute_panorama_ip_object_group_query(ip_address, device_group, vsys),
                                                timeout=max_timeout
                                            )
                                        )
                                    )
                                    result = future.result(timeout=max_timeout + 5)
                            except RuntimeError:
                                result = asyncio.run(
                                    asyncio.wait_for(
                                        execute_panorama_ip_object_group_query(ip_address, device_group, vsys),
                                        timeout=max_timeout
                                    )
                                )
                            except Exception as async_error:
                                print(f"DEBUG: Async execution error: {str(async_error)}", file=sys.stderr, flush=True)
                                import traceback
                                print(f"DEBUG: Async error traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                                result = {"error": f"Error executing query: {str(async_error)}"}
                        
                            # Display result - show LLM summary in table format
                            with st.chat_message("assistant"):
                                if isinstance(result, dict) and "error" in result:
                                    st.error(f"❌ {result['error']}")
                                elif isinstance(result, dict) and "ai_analysis" in result:
                                    # Display LLM summary (should be in table format)
                                    ai_analysis = result["ai_analysis"]
                                    if isinstance(ai_analysis, dict):
                                        summary = ai_analysis.get("summary")
                                        if summary:
                                            st.markdown(summary)
                                    elif isinstance(ai_analysis, str):
                                        st.markdown(ai_analysis)
                                else:
                                    # Fallback if no LLM analysis
                                    st.info("Query completed. No results found or analysis unavailable.")
                        
                            # Store result for persistence
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result
                            })
                            return
                        except asyncio.TimeoutError:
                            error_msg = "⏱️ Panorama query timed out. Please try again."
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                        except Exception as e:
                            error_msg = f"An error occurred: {str(e)}"
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                    elif selected_tool == "query_panorama_address_group_members":
                        # Handle Panorama address group members query
                        address_group_name = tool_params.get("address_group_name", "").strip()
                        device_group = tool_params.get("device_group")
                        vsys = tool_params.get("vsys", "vsys1")
                    
                        if not address_group_name:
                            error_msg = "Address group name not found in query. Please specify an address group name (e.g., 'what other IPs are in the address group leander_web')."
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                    
                        status_msg = f"🔎 Querying Panorama for members of address group **{address_group_name}**..."
                        with st.chat_message("assistant"):
                            st.info(status_msg)
                    
                        try:
                            max_timeout = 60
                            try:
                                asyncio.get_running_loop()
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(
                                        lambda: asyncio.run(
                                            asyncio.wait_for(
                                                execute_panorama_address_group_members_query(address_group_name, device_group, vsys),
                                                timeout=max_timeout
                                            )
                                        )
                                    )
                                    result = future.result(timeout=max_timeout + 5)
                            except RuntimeError:
                                result = asyncio.run(
                                    asyncio.wait_for(
                                        execute_panorama_address_group_members_query(address_group_name, device_group, vsys),
                                        timeout=max_timeout
                                    )
                                )
                            except Exception as async_error:
                                print(f"DEBUG: Async execution error: {str(async_error)}", file=sys.stderr, flush=True)
                                import traceback
                                print(f"DEBUG: Async error traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                                result = {"error": f"Error executing query: {str(async_error)}"}
                        
                            # Display result - show LLM summary in table format
                            with st.chat_message("assistant"):
                                if isinstance(result, dict) and "error" in result:
                                    st.error(f"❌ {result['error']}")
                                elif isinstance(result, dict) and "ai_analysis" in result:
                                    # Display LLM summary (should be in table format)
                                    ai_analysis = result["ai_analysis"]
                                    if isinstance(ai_analysis, dict):
                                        summary = ai_analysis.get("summary")
                                        if summary:
                                            st.markdown(summary)
                                    elif isinstance(ai_analysis, str):
                                        st.markdown(ai_analysis)
                                else:
                                    # Fallback if no LLM analysis
                                    st.info("Query completed. No results found or analysis unavailable.")
                        
                            # Store result for persistence
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result
                            })
                            return
                        except asyncio.TimeoutError:
                            error_msg = "⏱️ Panorama query timed out. Please try again."
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                        except Exception as e:
                            error_msg = f"An error occurred: {str(e)}"
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                    elif selected_tool == "get_splunk_recent_denies":
                        ip_address = tool_params.get("ip_address", "").strip()
                        limit = tool_params.get("limit", 100)
                        earliest_time = tool_params.get("earliest_time", "-24h")
                        if not ip_address:
                            error_msg = "IP address not found in query. Please specify an IP (e.g., 'get recent denies for 192.168.1.1')."
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                        status_msg = f"🔎 Querying Splunk for recent denies for **{ip_address}**..."
                        with st.chat_message("assistant"):
                            st.info(status_msg)
                        try:
                            max_timeout = 95
                            try:
                                asyncio.get_running_loop()
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(
                                        lambda: asyncio.run(
                                            asyncio.wait_for(
                                                execute_splunk_recent_denies_query(ip_address, limit, earliest_time),
                                                timeout=max_timeout
                                            )
                                        )
                                    )
                                    result = future.result(timeout=max_timeout + 5)
                            except RuntimeError:
                                result = asyncio.run(
                                    asyncio.wait_for(
                                        execute_splunk_recent_denies_query(ip_address, limit, earliest_time),
                                        timeout=max_timeout
                                    )
                                )
                            with st.chat_message("assistant"):
                                if isinstance(result, dict) and result.get("error"):
                                    st.error(f"❌ {result['error']}")
                                elif isinstance(result, dict):
                                    events = result.get("events", [])
                                    count = result.get("count", len(events))
                                    st.success(f"Found **{count}** recent deny event(s) for **{ip_address}** from Splunk.")
                                    if events:
                                        import pandas as pd
                                        df = pd.DataFrame(events)
                                        st.dataframe(df, width="stretch")
                                    else:
                                        st.info("No deny events found for this IP in the time range.")
                                else:
                                    st.info("Query completed. No results returned.")
                            st.session_state.messages.append({"role": "assistant", "content": result})
                            return
                        except asyncio.TimeoutError:
                            error_msg = "⏱️ Splunk query timed out. Please try again."
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                        except Exception as e:
                            error_msg = f"An error occurred: {str(e)}"
                            with st.chat_message("assistant"):
                                st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            return
                    else:
                        # Tool was selected but not handled - this shouldn't happen
                        print(f"DEBUG: WARNING - Tool '{selected_tool}' was selected but not handled!", file=sys.stderr, flush=True)
                        error_msg = f"Tool '{selected_tool}' is not yet implemented or handled."
                        with st.chat_message("assistant"):
                            st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        return
                else:
                    # Tool selection failed or no tool selected - show error, no fallback
                    error_msg = tool_selection.get('error', 'Failed to select a tool. Please rephrase your query.')
                    print(f"DEBUG: Tool selection failed. success={tool_selection.get('success')}, tool_name={tool_selection.get('tool_name')}, error={error_msg}", file=sys.stderr, flush=True)
                    with st.chat_message("assistant"):
                        st.error(f"❌ {error_msg}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    return
            except Exception as e:
                print(f"DEBUG: Tool discovery execution failed: {str(e)}", file=sys.stderr, flush=True)
                import traceback
                print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                # Show error, no fallback
                error_msg = f"Error processing query: {str(e)}"
                with st.chat_message("assistant"):
                    st.error(f"❌ {error_msg}")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                return
            # Store pending query for confirmation
            query_id = len(st.session_state.messages)
            pending_key = f"pending_query_{query_id}"
            
            # Check if this query already exists (from previous rerun)
            if pending_key not in st.session_state:
                st.session_state[pending_key] = {
                    'source': parsed['source'],
                    'destination': parsed['destination'],
                    'protocol': parsed['protocol'],
                    'port': parsed['port'],
                    'suggested_live': parsed['is_live']
                }
            
            # Check for button clicks first (before displaying buttons)
            button_click_key = f"button_clicked_{query_id}"
            if button_click_key in st.session_state:
                # Button was clicked, set confirmed
                data_type = st.session_state[button_click_key]
                is_live_choice = (data_type == "live")
                st.session_state[pending_key]['is_live'] = is_live_choice
                st.session_state[pending_key]['confirmed'] = True
                print(f"DEBUG: Button click detected from previous run: {data_type}, setting confirmed=True", file=sys.stderr, flush=True)
                # Clear the button click tracker
                del st.session_state[button_click_key]
                # Add confirmation message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Query confirmed. Using {data_type} data..."
                })
                st.rerun()
            
            # Ask user to confirm live data preference
            with st.chat_message("assistant"):
                st.info(f"📋 I found a path query from **{parsed['source']}** to **{parsed['destination']}** using **{parsed['protocol']}** port **{parsed['port']}**.")
                
                # Determine suggested live data setting
                suggested_live = parsed['is_live']
                if 'live' in prompt.lower() or 'baseline' in prompt.lower():
                    suggested_live = 'live' in prompt.lower() and 'baseline' not in prompt.lower()
                
                st.markdown("**Please choose the data source:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    use_live = st.button(
                        "🔴 Use Live Data",
                        key=f"live_btn_{query_id}",
                        width="stretch",
                        type="primary" if suggested_live else "secondary",
                        help="Use real-time live access data (may take longer but more current)"
                    )
                
                with col2:
                    use_baseline = st.button(
                        "💾 Use Cached/Baseline Data",
                        key=f"baseline_btn_{query_id}",
                        width="stretch",
                        type="primary" if not suggested_live else "secondary",
                        help="Use cached baseline data (faster but may be older)"
                    )
                
                # If user clicked a button, store the choice and trigger execution
                if use_live:
                    print(f"DEBUG: Live button clicked! pending_key={pending_key}", file=sys.stderr, flush=True)
                    st.session_state[button_click_key] = "live"
                    st.rerun()
                elif use_baseline:
                    print(f"DEBUG: Baseline button clicked! pending_key={pending_key}", file=sys.stderr, flush=True)
                    st.session_state[button_click_key] = "baseline"
                    st.rerun()

    # Check for confirmed pending queries to execute
    print(f"DEBUG: Checking for pending queries. Session state keys: {[k for k in st.session_state.keys() if k.startswith('pending_query_')]}", file=sys.stderr, flush=True)
    executed_query = False
    for key in list(st.session_state.keys()):
        if key.startswith('pending_query_') and isinstance(st.session_state[key], dict):
            query_data = st.session_state[key]
            confirmed = query_data.get('confirmed', False)
            
            # Also check if button was clicked using the button click tracker
            query_id = key.split('_')[-1]
            button_click_key = f"button_clicked_{query_id}"
            print(f"DEBUG: Checking for button click tracker: {button_click_key}", file=sys.stderr, flush=True)
            print(f"DEBUG: All session state keys: {[k for k in st.session_state.keys() if 'button' in k.lower() or 'clicked' in k.lower()]}", file=sys.stderr, flush=True)
            
            if button_click_key in st.session_state:
                print(f"DEBUG: Button click detected via tracker: {st.session_state[button_click_key]}", file=sys.stderr, flush=True)
                # Set confirmed based on button click
                confirmed = True
                query_data['confirmed'] = True
                if st.session_state[button_click_key] == "live":
                    query_data['is_live'] = True
                else:
                    query_data['is_live'] = False
                # Clear the tracker
                del st.session_state[button_click_key]
            
            print(f"DEBUG: Found pending query {key}, confirmed={confirmed}, executed_query={executed_query}", file=sys.stderr, flush=True)
            if confirmed and not executed_query:
                # Copy query data before deleting
                source = query_data['source']
                destination = query_data['destination']
                protocol = query_data['protocol']
                port = query_data['port']
                is_live = query_data['is_live']
                
                # Remove from pending
                del st.session_state[key]
                executed_query = True
                
                # Execute the query
                print(f"DEBUG: Starting query execution for {source} -> {destination}", file=sys.stderr, flush=True)
                
                # Display status message (temporary, not stored in message history)
                data_type = "live" if is_live else "baseline"
                status_msg = f"🔍 Querying network path using {data_type} data... This may take a moment."
                with st.chat_message("assistant"):
                    st.info(status_msg)
                
                try:
                    print(f"DEBUG: About to execute async query", file=sys.stderr, flush=True)
                    # Execute query with timeout
                    # Use asyncio.wait_for to add a timeout (5 minutes max for live data)
                    max_timeout = 300 if is_live else 120  # 5 min for live, 2 min for baseline
                    
                    # Check if there's already an event loop running (Streamlit might have one)
                    try:
                        loop = asyncio.get_running_loop()
                        # If we're already in an async context, we can't use asyncio.run()
                        # Instead, create a task
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                lambda: asyncio.run(
                                    asyncio.wait_for(
                                        execute_network_query(
                                            source,
                                            destination,
                                            protocol,
                                            port,
                                            is_live
                                        ),
                                        timeout=max_timeout
                                    )
                                )
                            )
                            result = future.result(timeout=max_timeout + 10)
                    except RuntimeError:
                        # No event loop running, we can use asyncio.run()
                        result = asyncio.run(
                            asyncio.wait_for(
                                execute_network_query(
                                    source,
                                    destination,
                                    protocol,
                                    port,
                                    is_live
                                ),
                                timeout=max_timeout
                            )
                        )
                    except Exception as e:
                        print(f"DEBUG: Exception during inner query execution: {e}", file=sys.stderr, flush=True)
                        result = {"error": f"Error executing query: {str(e)}"}
                    
                    print(f"DEBUG: Query execution completed, result type: {type(result)}", file=sys.stderr, flush=True)
                    
                    if result:
                        # Debug: Print result keys to help diagnose
                        if isinstance(result, dict):
                            print(f"DEBUG: Result keys: {list(result.keys())}", file=sys.stderr, flush=True)
                            print(f"DEBUG: Result sample: {str(result)[:500]}", file=sys.stderr, flush=True)
                        else:
                            print(f"DEBUG: Result is not a dict: {type(result)}", file=sys.stderr, flush=True)
                        
                        # Check if result contains an error
                        if isinstance(result, dict) and 'error' in result:
                            print(f"DEBUG: Result contains error: {result['error']}", file=sys.stderr, flush=True)
                            
                            with st.chat_message("assistant"):
                                st.error(f"❌ {result['error']}")
                                
                                # Extract and display statusDescription if available
                                if 'details' in result:
                                    details = result['details']
                                    if isinstance(details, str):
                                        # Try to extract statusDescription from details string
                                        if 'statusDescription:' in details:
                                            try:
                                                # Extract the status description
                                                desc_start = details.find('statusDescription:') + len('statusDescription:')
                                                desc_text = details[desc_start:].strip()
                                                # Remove any trailing statusCode or other info
                                                if ',' in desc_text:
                                                    desc_text = desc_text.split(',')[0].strip()
                                                if desc_text and desc_text != 'No description':
                                                    st.warning(f"ℹ️ {desc_text}")
                                            except:
                                                pass
                                        # If details is JSON-like, try to parse it
                                        elif details.startswith('{') or 'statusCode' in details:
                                            st.info(f"Details: {details}")
                                    elif isinstance(details, dict):
                                        status_desc = details.get('statusDescription', '')
                                        if status_desc and status_desc != 'No description':
                                            st.warning(f"ℹ️ {status_desc}")
                                
                                # Show source IP if available
                                if 'source' in result:
                                    st.info(f"Source: {result['source']}")
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result
                            })
                        else:
                            # Check if this is a Panorama result
                            if isinstance(result, dict) and "ip_address" in result and ("address_objects" in result or "error" in result):
                                print(f"DEBUG: Displaying Panorama result", file=sys.stderr, flush=True)
                                with st.chat_message("assistant"):
                                    if "error" in result:
                                        st.error(f"❌ {result['error']}")
                                    elif "ai_analysis" in result:
                                        # Display LLM summary (should be in table format)
                                        ai_analysis = result["ai_analysis"]
                                        if isinstance(ai_analysis, dict):
                                            summary = ai_analysis.get("summary")
                                            if summary:
                                                st.markdown(summary)
                                        elif isinstance(ai_analysis, str):
                                            st.markdown(ai_analysis)
                                    else:
                                        st.info("Query completed. No results found or analysis unavailable.")
                            else:
                                print(f"DEBUG: Displaying result using display_result_chat", file=sys.stderr, flush=True)
                                if isinstance(result, dict):
                                    print(f"DEBUG: Result keys: {list(result.keys())}", file=sys.stderr, flush=True)
                                    print(f"DEBUG: Has path_hops: {'path_hops' in result}, Has simplified_hops: {'simplified_hops' in result}", file=sys.stderr, flush=True)
                                    if 'path_hops' in result:
                                        print(f"DEBUG: path_hops type: {type(result['path_hops'])}, length: {len(result['path_hops']) if result['path_hops'] else 0}", file=sys.stderr, flush=True)
                                    if 'simplified_hops' in result:
                                        print(f"DEBUG: simplified_hops type: {type(result['simplified_hops'])}, length: {len(result['simplified_hops']) if result['simplified_hops'] else 0}", file=sys.stderr, flush=True)
                                # Display result in a new chat message
                                with st.chat_message("assistant"):
                                    try:
                                        display_result_chat(result, st.container())
                                        print(f"DEBUG: display_result_chat completed", file=sys.stderr, flush=True)
                                    except Exception as display_error:
                                        print(f"DEBUG: Error in display_result_chat: {display_error}", file=sys.stderr, flush=True)
                                        import traceback
                                        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                                        # Fallback: show as JSON
                                        st.json(result)
                            
                            # Add to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result
                            })
                            print(f"DEBUG: Result added to chat history", file=sys.stderr, flush=True)
                    else:
                        print(f"DEBUG: Result is None or empty", file=sys.stderr, flush=True)
                        
                        with st.chat_message("assistant"):
                            st.warning("No results returned from the query.")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "No results returned from the query."
                        })
                except asyncio.TimeoutError:
                    error_msg = f"⏱️ Query timed out after {max_timeout} seconds. The network path calculation is taking longer than expected. Please try again or use baseline data instead of live data."
                    with st.chat_message("assistant"):
                        st.error(error_msg)
                        st.info("💡 Tip: Try using baseline data instead of live data for faster results, or check if the NetBrain server is responding.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    print(f"DEBUG: Exception during query execution: {error_msg}", file=sys.stderr, flush=True)
                    import traceback
                    print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                    with st.chat_message("assistant"):
                        st.error(error_msg)
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                # Break after executing one query to avoid multiple executions
                break


# Check if this script is being run directly (not imported as a module)
# __name__ will be "__main__" when the script is executed directly
# This allows the script to be both runnable and importable
if __name__ == "__main__":
    # Call the main function to start the Streamlit application
    # This will launch the web interface when the script is run
    main()
