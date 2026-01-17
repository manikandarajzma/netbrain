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
    
    # Add source node
    if source:
        devices.add(source)
    
    # Process each hop
    for hop in path_hops:
        from_dev = hop.get('from_device', 'Unknown')
        to_dev = hop.get('to_device')
        status = hop.get('status', 'Unknown')
        failure_reason = hop.get('failure_reason')
        
        if from_dev and from_dev != 'Unknown':
            devices.add(from_dev)
        if to_dev:
            devices.add(to_dev)
        
        # Determine edge color and style based on status
        if status == 'Failed' or failure_reason:
            edge_color = 'red'
            edge_style = 'dashed'
            edge_width = 2.5
        elif status == 'Success':
            edge_color = 'green'
            edge_style = 'solid'
            edge_width = 2.0
        else:
            edge_color = 'gray'
            edge_style = 'solid'
            edge_width = 1.5
        
        # Add edge with attributes
        if from_dev and to_dev:
            edges.append((from_dev, to_dev, {
                'color': edge_color,
                'style': edge_style,
                'width': edge_width,
                'status': status,
                'failure_reason': failure_reason
            }))
        elif from_dev and not to_dev:
            # Last hop - connect to destination
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
    
    # Add nodes and edges to graph
    G.add_nodes_from(devices)
    for from_dev, to_dev, attrs in edges:
        G.add_edge(from_dev, to_dev, **attrs)
    
    if len(G.nodes()) == 0:
        return None
    
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
            
            # Position intermediate devices
            for hop in path_hops:
                from_dev = hop.get('from_device', 'Unknown')
                to_dev = hop.get('to_device')
                
                if from_dev and from_dev != 'Unknown' and from_dev not in pos:
                    pos[from_dev] = (x_pos, y_center)
                    x_pos += 2
                
                if to_dev and to_dev not in pos:
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
    
    # Draw labels
    labels = {node: node[:15] + '...' if len(node) > 15 else node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Source'),
        Patch(facecolor='#2196F3', label='Intermediate Device'),
        Patch(facecolor='#FF9800', label='Destination'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Success'),
        plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Failed'),
        plt.Line2D([0], [0], color='gray', linewidth=1.5, label='Unknown')
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
                st.error("Error Details:")
                st.json(result['details'])
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
                        st.info(f"Gateway Used: {result['gateway_used']}")
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
                        
                        if from_dev_name != 'Unknown' or to_dev_name:
                            hop_info = {
                                'from_device': from_dev_name,
                                'to_device': to_dev_name,
                                'status': branch_status,
                                'failure_reason': branch_failure_reason
                            }
                            simplified_hops.append(hop_info)
        
        return simplified_hops if simplified_hops else None
    except Exception as e:
        print(f"DEBUG: Error extracting hops from path_details: {e}", file=sys.stderr, flush=True)
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return None

def display_result_chat(result, container):
    """
    Display network path query result in a chat message container.
    
    Args:
        result: Dictionary containing the query result from the MCP server
        container: Streamlit container to display results in
    """
    print(f"DEBUG: display_result_chat called with result type: {type(result)}", file=sys.stderr, flush=True)
    if isinstance(result, dict):
        print(f"DEBUG: Result is dict, checking for error key", file=sys.stderr, flush=True)
        if 'error' in result:
            container.error(f"❌ Error: {result['error']}")
            if 'details' in result:
                with container.expander("Error Details"):
                    container.json(result['details'])
        else:
            print(f"DEBUG: No error in result, checking path status", file=sys.stderr, flush=True)
            # Check if path calculation was successful or failed
            path_status = result.get('path_status', 'Unknown')
            path_status_description = result.get('path_status_description', '')
            path_failure_reason = result.get('path_failure_reason', '')
            
            path_failed = (
                path_status == 'Failed' or 
                'Failed' in str(path_status) or 
                'failed' in str(path_status_description).lower() or
                path_failure_reason
            )
            
            if path_failed:
                container.error("❌ Network Path Calculation Failed")
                # Show status
                if path_status:
                    container.error(f"**Status:** {path_status}")
                elif path_status_description:
                    container.error(f"**Status:** {path_status_description}")
                # Show failure reason
                if path_failure_reason:
                    container.error(f"**Reason:** {path_failure_reason}")
            
            # Check for path hops in multiple possible locations
            hops_to_display = None
            if 'path_hops' in result and result['path_hops']:
                hops_to_display = result['path_hops']
                print(f"DEBUG: Using path_hops, count: {len(hops_to_display)}", file=sys.stderr, flush=True)
            elif 'simplified_hops' in result and result['simplified_hops']:
                hops_to_display = result['simplified_hops']
                print(f"DEBUG: Using simplified_hops, count: {len(hops_to_display)}", file=sys.stderr, flush=True)
            elif 'path_details' in result and result['path_details']:
                # Try to extract hops from path_details as fallback
                print(f"DEBUG: Attempting to extract hops from path_details", file=sys.stderr, flush=True)
                hops_to_display = extract_hops_from_path_details(result['path_details'])
                if hops_to_display:
                    print(f"DEBUG: Extracted {len(hops_to_display)} hops from path_details", file=sys.stderr, flush=True)
                else:
                    print(f"DEBUG: Could not extract hops from path_details", file=sys.stderr, flush=True)
            
            # Display path hops if available (always show path details)
            if hops_to_display:
                container.markdown("### Network Path")
                
                # Create and display graph visualization
                print(f"DEBUG: Creating graph with {len(hops_to_display)} hops", file=sys.stderr, flush=True)
                try:
                    graph_fig = create_path_graph(
                        hops_to_display,
                        result.get('source', 'Source'),
                        result.get('destination', 'Destination')
                    )
                    if graph_fig:
                        print(f"DEBUG: Graph created successfully", file=sys.stderr, flush=True)
                        container.markdown("#### Path Visualization")
                        container.pyplot(graph_fig)
                        plt.close(graph_fig)
                    else:
                        print(f"DEBUG: Graph creation returned None", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"DEBUG: Graph generation exception: {e}", file=sys.stderr, flush=True)
                    import traceback
                    print(f"DEBUG: Graph traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                    container.warning(f"Could not generate graph: {str(e)}")
                
                # Display hops
                container.markdown("#### Path Hops")
                for i, hop in enumerate(hops_to_display):
                    from_dev = hop.get('from_device', 'Unknown')
                    to_dev = hop.get('to_device', 'Destination')
                    status = hop.get('status', 'Unknown')
                    failure_reason = hop.get('failure_reason')
                    
                    if status == 'Failed' or failure_reason:
                        container.error(f"**{from_dev}** → **{to_dev}**: ❌ {status}")
                        if failure_reason:
                            container.caption(f"Reason: {failure_reason}")
                    elif status == 'Success':
                        container.success(f"**{from_dev}** → **{to_dev}**: ✓ {status}")
                    else:
                        container.info(f"**{from_dev}** → **{to_dev}**: {status}")
                
                with container.expander("View Full Details"):
                    container.json(result)
            else:
                # Show summary information even without path_hops
                # Check if path failed first
                path_failed_summary = (
                    result.get('path_status') == 'Failed' or 
                    'Failed' in str(result.get('path_status', '')) or 
                    'failed' in str(result.get('path_status_description', '')).lower() or
                    result.get('path_failure_reason') or result.get('failure_reason') or
                    (result.get('statusCode') and result.get('statusCode') != 790200)
                )
                
                if path_failed_summary:
                    # For failures, show only status and failure reason
                    container.error("❌ Network Path Calculation Failed")
                    
                    # Show status
                    path_status = result.get('path_status', 'Failed')
                    if path_status and path_status != 'Unknown':
                        container.error(f"**Status:** {path_status}")
                    elif result.get('statusCode'):
                        status_code = result.get('statusCode')
                        status_desc = result.get('statusDescription', '')
                        if status_desc:
                            container.error(f"**Status:** {status_desc}")
                        else:
                            container.error(f"**Status:** Failed (Status Code: {status_code})")
                    
                    # Show failure reason
                    failure_reason = result.get('path_failure_reason') or result.get('failure_reason')
                    if failure_reason:
                        container.error(f"**Reason:** {failure_reason}")
                else:
                    # For successful queries without path_hops, check for simplified_hops
                    # Check if we have simplified_hops or any path data
                    hops_to_display = None
                    if 'simplified_hops' in result and result['simplified_hops']:
                        hops_to_display = result['simplified_hops']
                    elif 'path_hops' in result and result['path_hops']:
                        hops_to_display = result['path_hops']
                    
                    if hops_to_display:
                        # Display path with graph visualization
                        container.markdown("### Network Path")
                        
                        # Create and display graph visualization
                        try:
                            graph_fig = create_path_graph(
                                hops_to_display,
                                result.get('source', 'Source'),
                                result.get('destination', 'Destination')
                            )
                            if graph_fig:
                                container.markdown("#### Path Visualization")
                                container.pyplot(graph_fig)
                                plt.close(graph_fig)
                        except Exception as e:
                            container.warning(f"Could not generate graph: {str(e)}")
                            import traceback
                            print(f"DEBUG: Graph generation error: {traceback.format_exc()}", file=sys.stderr, flush=True)
                        
                        # Display hops
                        container.markdown("#### Path Hops")
                        for hop in hops_to_display:
                            from_dev = hop.get('from_device', 'Unknown')
                            to_dev = hop.get('to_device', 'Destination')
                            status = hop.get('status', 'Unknown')
                            failure_reason = hop.get('failure_reason')
                            
                            if status == 'Failed' or failure_reason:
                                container.error(f"**{from_dev}** → **{to_dev}**: ❌ {status}")
                                if failure_reason:
                                    container.caption(f"Reason: {failure_reason}")
                            elif status == 'Success':
                                container.success(f"**{from_dev}** → **{to_dev}**: ✓ {status}")
                            else:
                                container.info(f"**{from_dev}** → **{to_dev}**: {status}")
                    else:
                        # No path data available, show basic info
                        container.info("✅ Path calculation completed successfully")
                        if 'taskID' in result:
                            container.info(f"📋 Task ID: {result['taskID']}")
                        if 'gateway_used' in result:
                            container.info(f"🔌 Gateway: {result['gateway_used']}")
                        # Show full details in expander for debugging
                        with container.expander("View Full Response Details"):
                            container.json(result)
    else:
        # Fallback: display any result type
        print(f"DEBUG: Result is not a dict, displaying as JSON", file=sys.stderr, flush=True)
        container.json(result)
    
    print(f"DEBUG: display_result_chat completed", file=sys.stderr, flush=True)

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
        
        # Debug section
        with st.expander("🔍 Debug Info"):
            pending_keys = [k for k in st.session_state.keys() if k.startswith('pending_query_')]
            st.write(f"Pending queries: {len(pending_keys)}")
            for key in pending_keys:
                query_data = st.session_state[key]
                st.json({key: query_data})
        
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
