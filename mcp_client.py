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

# Import ClientSession for managing MCP client connections
# Import StdioServerParameters for configuring stdio-based server communication
# Import stdio_client helper function for creating stdio transport streams
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import ChatOllama for LLM integration (currently imported but not actively used in client)
from langchain_ollama import ChatOllama

# Import pandas for reading spreadsheet files (CSV, Excel)
import pandas as pd

# Import json for serialization
import json

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
    
    # If it's already a string, return it
    if isinstance(interface_data, str):
        return interface_data
    
    # If it's a dictionary, try to extract the interface name
    if isinstance(interface_data, dict):
        # Try PhysicalInftName first (as seen in the UI)
        if 'PhysicalInftName' in interface_data:
            return interface_data['PhysicalInftName']
        # Try intfDisplaySchemaObj.value
        if 'intfDisplaySchemaObj' in interface_data and isinstance(interface_data['intfDisplaySchemaObj'], dict):
            if 'value' in interface_data['intfDisplaySchemaObj']:
                return interface_data['intfDisplaySchemaObj']['value']
        # Try common field names
        for field in ['name', 'interface', 'intf', 'interfaceName', 'intfName']:
            if field in interface_data:
                return interface_data[field]
    
    # If we can't extract it, return None
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
    
    # Track firewall interfaces, zones, and device groups for overlay
    firewall_interfaces = {}  # {device_name: {'in': 'interface', 'out': 'interface', 'in_zone': 'zone', 'out_zone': 'zone', 'device_group': 'group'}}
    
    # Add source node
    if source:
        devices.add(source)
        print(f"DEBUG: Added source node: {source}", file=sys.stderr, flush=True)
    
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
                
                # Use extract_interface_name helper to get clean interface names
                in_intf_name = extract_interface_name(in_interface) if in_interface else None
                out_intf_name = extract_interface_name(out_interface) if out_interface else None
                
                # Store interface, zone, and device group information for this firewall
                if firewall_device not in firewall_interfaces:
                    firewall_interfaces[firewall_device] = {'in': None, 'out': None, 'in_zone': None, 'out_zone': None, 'device_group': None}
                
                # Update interfaces if we have new information
                if in_intf_name and not firewall_interfaces[firewall_device]['in']:
                    firewall_interfaces[firewall_device]['in'] = in_intf_name
                if out_intf_name and not firewall_interfaces[firewall_device]['out']:
                    firewall_interfaces[firewall_device]['out'] = out_intf_name
                
                # Update zones if we have new information (use 'or' to allow overwriting None)
                if in_zone:
                    firewall_interfaces[firewall_device]['in_zone'] = in_zone
                    print(f"DEBUG: Graph - Set in_zone for {firewall_device} to {in_zone}", file=sys.stderr, flush=True)
                if out_zone:
                    firewall_interfaces[firewall_device]['out_zone'] = out_zone
                    print(f"DEBUG: Graph - Set out_zone for {firewall_device} to {out_zone}", file=sys.stderr, flush=True)
                if device_group:
                    firewall_interfaces[firewall_device]['device_group'] = device_group
                    print(f"DEBUG: Graph - Set device_group for {firewall_device} to {device_group}", file=sys.stderr, flush=True)
        
        # Add valid devices
        if from_dev and from_dev != 'Unknown':
            devices.add(from_dev)
        if to_dev and to_dev not in [None, 'None', 'null', '']:
            devices.add(to_dev)
        
        # Determine edge color and style based on status
        if status == 'Success':
            edge_color = 'green'
            edge_style = 'solid'
            edge_width = 2.0
        elif status == 'Failed' or failure_reason:
            # Show failed hops with red/dashed style so users can see what was discovered
            edge_color = 'red'
            edge_style = 'dashed'
            edge_width = 1.5
        else:
            edge_color = 'gray'
            edge_style = 'solid'
            edge_width = 1.5
        
        # Add edge with attributes - for all valid device pairs (including failed ones)
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
            # Last hop - connect to destination if available
            if destination:
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
    
    # Add destination node if not already added
    if destination and destination not in devices:
        devices.add(destination)
        print(f"DEBUG: Added destination node: {destination}", file=sys.stderr, flush=True)
    
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use hierarchical layout for better path visualization
    try:
        # Try to create a hierarchical layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        # If we have a clear path, try to arrange it linearly
        if len(path_hops) > 0:
            # Create a more linear layout for path visualization
            pos = {}
            x_pos = 0
            y_center = 0
            
            # Position source
            if source and source in G.nodes():
                pos[source] = (x_pos, y_center)
                x_pos += 2
            
            # Position intermediate devices - only valid ones
            for hop in path_hops:
                from_dev = hop.get('from_device', 'Unknown')
                to_dev = hop.get('to_device')
                
                # Skip invalid devices
                if from_dev and from_dev not in [None, 'None', 'null', ''] and from_dev != 'Unknown' and from_dev not in pos:
                    pos[from_dev] = (x_pos, y_center)
                    x_pos += 2
                
                if to_dev and to_dev not in [None, 'None', 'null', ''] and to_dev not in pos:
                    pos[to_dev] = (x_pos, y_center)
                    x_pos += 2
            
            # Position destination
            if destination and destination not in pos:
                pos[destination] = (x_pos, y_center)
            
            # Fill in any missing positions
            for node in G.nodes():
                if node not in pos:
                    pos[node] = (x_pos, y_center)
                    x_pos += 2
    except:
        # Fallback to spring layout
        pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node == source:
            node_colors.append('#4CAF50')  # Green for source
        elif node == destination:
            node_colors.append('#FF9800')  # Orange for destination
        else:
            node_colors.append('#2196F3')  # Blue for intermediate devices
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, 
                           alpha=0.9, ax=ax)
    
    # Draw edges with colors and styles
    for from_dev, to_dev, data in G.edges(data=True):
        edge_color = data.get('color', 'gray')
        edge_style = data.get('style', 'solid')
        edge_width = data.get('width', 1.5)
        
        nx.draw_networkx_edges(G, pos, edgelist=[(from_dev, to_dev)], 
                              edge_color=edge_color, style=edge_style,
                              width=edge_width, alpha=0.7, arrows=True,
                              arrowsize=20, ax=ax, connectionstyle='arc3,rad=0.1')
    
    # Draw labels with device type information, positioned slightly above nodes
    print(f"DEBUG: Graph - Device types collected: {device_types}", file=sys.stderr, flush=True)
    label_offset = 0.015  # Offset to position labels above nodes (reduced further to bring closer)
    for node in G.nodes():
        if node in pos:
            x, y = pos[node]
            node_label = node[:15] + '...' if len(node) > 15 else node
            # Add device type if available
            if node in device_types and device_types[node]:
                device_type = device_types[node]
                # Format device type nicely (e.g., "Palo Alto Firewall" -> "Palo Alto Firewall")
                label_text = f"{node_label}\n({device_type})"
                print(f"DEBUG: Graph - Label for '{node}': '{node_label}\\n({device_type})'", file=sys.stderr, flush=True)
            else:
                label_text = node_label
                print(f"DEBUG: Graph - No device type for '{node}', using label: '{node_label}'", file=sys.stderr, flush=True)
            
            # Draw label slightly above the node
            ax.text(x, y + label_offset, label_text, 
                   fontsize=7, fontweight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    # Overlay firewall interface and zone information
    for firewall_device, interfaces in firewall_interfaces.items():
        if firewall_device in pos:
            x, y = pos[firewall_device]
            
            # Debug: Print what we have for this firewall
            print(f"DEBUG: Graph overlay for {firewall_device}: interfaces={interfaces}", file=sys.stderr, flush=True)
            
            # Calculate vertical positions for labels
            # Cleaner layout: interface+zone combined on top/bottom, device group in middle
            top_offset = 0.02  # Top label position (reduced further to bring closer)
            bottom_offset = 0.02  # Bottom label position (reduced further to bring closer)
            device_group_offset = 0.005  # Device group label position (slightly below center, reduced further)
            
            # Build top label: In interface with zone
            top_label_parts = []
            if interfaces.get('in'):
                in_zone = interfaces.get('in_zone')
                if in_zone:
                    top_label_parts.append(f"In: {interfaces['in']} ({in_zone})")
                else:
                    top_label_parts.append(f"In: {interfaces['in']}")
            
            # Build bottom label: Out interface with zone
            bottom_label_parts = []
            if interfaces.get('out'):
                out_zone = interfaces.get('out_zone')
                print(f"DEBUG: {firewall_device} - Out zone: {out_zone}", file=sys.stderr, flush=True)
                if out_zone:
                    bottom_label_parts.append(f"Out: {interfaces['out']} ({out_zone})")
                else:
                    bottom_label_parts.append(f"Out: {interfaces['out']}")
            
            # Add top label (In interface + zone)
            if top_label_parts:
                ax.text(x, y + top_offset, top_label_parts[0], 
                       fontsize=6, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
            
            # Add bottom label (Out interface + zone)
            if bottom_label_parts:
                ax.text(x, y - bottom_offset, bottom_label_parts[0], 
                       fontsize=6, ha='center', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            
            # Add device group label in the middle (if available)
            device_group = interfaces.get('device_group')
            if device_group:
                ax.text(x, y - device_group_offset, f"DG: {device_group}", 
                       fontsize=6, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.9),
                       weight='bold')
    
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

def get_server_params():
    """
    Create server parameters for stdio communication.
    
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
    
    # Only skip if no path data AND failure reason is L2 connection discovery
    if not hops_to_display:
        path_failure_reason = result.get('path_failure_reason', '')
        if path_failure_reason and 'L2 connections has not been discovered' in path_failure_reason:
            print(f"DEBUG: Skipping display - no path data and L2 connection discovery issue", file=sys.stderr, flush=True)
            return
        print(f"DEBUG: No path hops found in result", file=sys.stderr, flush=True)
    
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
                    
                    # Extract interface names (handle both string and dict formats)
                    in_intf_name = extract_interface_name(in_interface)
                    out_intf_name = extract_interface_name(out_interface)
                    
                    firewalls_found[firewall_device] = {
                        'in_interface': in_intf_name,
                        'out_interface': out_intf_name,
                        'in_zone': in_zone,
                        'out_zone': out_zone,
                        'device_group': device_group
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
                    
                    if in_intf_name and not firewalls_found[firewall_device]['in_interface']:
                        firewalls_found[firewall_device]['in_interface'] = in_intf_name
                    if out_intf_name and not firewalls_found[firewall_device]['out_interface']:
                        firewalls_found[firewall_device]['out_interface'] = out_intf_name
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
        
        # Display firewall information in the requested format
        if firewalls_found:
            print(f"DEBUG: Found {len(firewalls_found)} firewalls", file=sys.stderr, flush=True)
            container.markdown("#### Firewalls in Path")
            for fw_name, fw_info in firewalls_found.items():
                interface_parts = []
                if fw_info['in_interface']:
                    in_zone = fw_info.get('in_zone')
                    zone_text = f" ({in_zone})" if in_zone else ""
                    interface_parts.append(f"In: {fw_info['in_interface']}{zone_text}")
                if fw_info['out_interface']:
                    out_zone = fw_info.get('out_zone')
                    zone_text = f" ({out_zone})" if out_zone else ""
                    interface_parts.append(f"Out: {fw_info['out_interface']}{zone_text}")
                
                if interface_parts:
                    device_group = fw_info.get('device_group')
                    device_group_text = f" [DG: {device_group}]" if device_group else ""
                    container.markdown(f"{fw_name}{device_group_text}: {', '.join(interface_parts)}")
        else:
            print(f"DEBUG: No firewalls found in path", file=sys.stderr, flush=True)
    else:
        print(f"DEBUG: No hops to display - result may not have path data", file=sys.stderr, flush=True)
        print(f"DEBUG: Result keys available: {list(result.keys())}", file=sys.stderr, flush=True)

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
        server_params = get_server_params()
        print(f"DEBUG: Server params created, connecting to MCP server...", file=sys.stderr, flush=True)
        
        async with stdio_client(server_params) as (read_stream, write_stream):
            print(f"DEBUG: Connected to MCP server, initializing session...", file=sys.stderr, flush=True)
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print(f"DEBUG: Session initialized, calling tool...", file=sys.stderr, flush=True)
                
                tool_arguments = {
                    "source": source,
                    "destination": destination,
                    "protocol": protocol,
                    "port": port,
                    "is_live": 1 if is_live else 0,
                    "continue_on_policy_denial": True  # Always continue even if denied by policy
                }
                print(f"DEBUG: Tool arguments: {tool_arguments}", file=sys.stderr, flush=True)
                
                tool_result = await session.call_tool(
                    "query_network_path",
                    arguments=tool_arguments
                )
                print(f"DEBUG: Tool call completed, processing result...", file=sys.stderr, flush=True)
                
                if tool_result and tool_result.content:
                    import json
                    result_text = tool_result.content[0].text
                    print(f"DEBUG: Result text length: {len(result_text)}", file=sys.stderr, flush=True)
                    try:
                        result = json.loads(result_text)
                        print(f"DEBUG: Result parsed successfully", file=sys.stderr, flush=True)
                        return result
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: JSON decode error: {e}", file=sys.stderr, flush=True)
                        return {"result": result_text}
                else:
                    print(f"DEBUG: No result content returned", file=sys.stderr, flush=True)
                    return None
    except asyncio.TimeoutError:
        print(f"DEBUG: Query timed out", file=sys.stderr, flush=True)
        return {"error": "Query timed out. The network path calculation is taking longer than expected."}
    except Exception as e:
        import traceback
        print(f"DEBUG: Exception in execute_network_query: {e}", file=sys.stderr, flush=True)
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": f"Error executing query: {str(e)}"}

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
    st.title("🌐 NetBrain Network Assistant")
    st.markdown("Ask me about network paths! Try: *'Find path from 10.0.0.1 to 10.0.1.1 using TCP port 80'*")
    
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
        - *Find path from 10.0.0.1 to 10.0.1.1*
        - *Query path from 192.168.1.10 to 192.168.2.20 using TCP port 443*
        - *Show me the network path from 10.10.3.253 to 172.24.32.225 UDP port 53 with live data*
        - *Check path from source 10.0.0.254 to destination 10.0.1.254 port 80*
        """)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your NetBrain Network Assistant. I can help you query network paths between devices. Just tell me the source and destination IPs, and optionally the protocol and port. For example: *'Find path from 10.0.0.1 to 10.0.1.1'*"
            }
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                # Assistant messages can contain various content types
                if isinstance(message["content"], dict):
                    # Display result
                    display_result_chat(message["content"], st.container())
                else:
                    st.markdown(message["content"])
    
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
                            use_container_width=True,
                            type="primary" if suggested_live else "secondary",
                            help="Use real-time live access data (may take longer but more current)"
                        )
                    
                    with col2:
                        use_baseline = st.button(
                            "💾 Use Cached/Baseline Data",
                            key=f"baseline_btn_{query_id}",
                            use_container_width=True,
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
        
        # Parse query
        default_live = st.session_state.get('default_live_data', True)
        parsed = parse_query(prompt, default_live)
        
        # Check if we have required information
        if not parsed['source'] or not parsed['destination']:
            with st.chat_message("assistant"):
                st.warning("⚠️ I need both source and destination IP addresses to query the network path. Please provide both in your query.")
                st.info("Example: *'Find path from 10.0.0.1 to 10.0.1.1'*")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I need both source and destination IP addresses. Please provide both in your query."
                })
        else:
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
                        use_container_width=True,
                        type="primary" if suggested_live else "secondary",
                        help="Use real-time live access data (may take longer but more current)"
                    )
                
                with col2:
                    use_baseline = st.button(
                        "💾 Use Cached/Baseline Data",
                        key=f"baseline_btn_{query_id}",
                        use_container_width=True,
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
                
                # Add status message to chat history first
                data_type = "live" if is_live else "baseline"
                status_msg = f"🔍 Querying network path using {data_type} data... This may take a moment."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": status_msg
                })
                
                # Display status message
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
                    
                    # Remove status message from history
                    if st.session_state.messages and st.session_state.messages[-1]["content"] == status_msg:
                        st.session_state.messages.pop()
                    
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
                            # Remove status message from history
                            if st.session_state.messages and st.session_state.messages[-1]["content"] == status_msg:
                                st.session_state.messages.pop()
                            
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
                        # Remove status message from history
                        if st.session_state.messages and st.session_state.messages[-1]["content"] == status_msg:
                            st.session_state.messages.pop()
                        
                        with st.chat_message("assistant"):
                            st.warning("No results returned from the query.")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "No results returned from the query."
                        })
                except asyncio.TimeoutError:
                    # Remove status message from history
                    if st.session_state.messages and st.session_state.messages[-1]["content"] == status_msg:
                        st.session_state.messages.pop()
                    
                    error_msg = f"⏱️ Query timed out after {max_timeout} seconds. The network path calculation is taking longer than expected. Please try again or use baseline data instead of live data."
                    with st.chat_message("assistant"):
                        st.error(error_msg)
                        st.info("💡 Tip: Try using baseline data instead of live data for faster results, or check if the NetBrain server is responding.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                except Exception as e:
                    # Remove status message from history
                    if st.session_state.messages and st.session_state.messages[-1]["content"] == status_msg:
                        st.session_state.messages.pop()
                    
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
