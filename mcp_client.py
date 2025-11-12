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
            # Display success message in green using st.success()
            st.success("Query completed successfully")
            
            # Display path hops if available (simplified visual representation)
            if 'path_hops' in result:
                st.subheader("Network Path")
                
                # Display path status
                path_status = result.get('path_status_description', result.get('path_status', 'Unknown'))
                if 'Failed' in str(path_status) or result.get('path_status') != 790200:
                    st.error(f"Path Status: {path_status}")
                else:
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
                # Display the result dictionary as formatted JSON
                # st.json() pretty-prints JSON with syntax highlighting
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
    # Create server parameters for stdio communication:
    # - command: The command to run (python interpreter)
    # - args: Arguments to pass (the mcp_server.py script)
    # This configures the client to spawn mcp_server.py as a subprocess
    return StdioServerParameters(
        command="python",  # Use Python interpreter to run the server
        args=["mcp_server.py"]  # Pass mcp_server.py as argument
    )

def main():
    """
    Main function that creates and manages the Streamlit web interface.
    
    This function:
    1. Creates a form with input fields for network query parameters
    2. Validates user input
    3. Calls the MCP server tool to query network paths
    4. Displays results or error messages to the user
    """
    # Display the main page title as a large heading
    st.title("NetBrain Network Query")
    
    # Create tabs to switch between natural language, form input, and spreadsheet upload
    tab1, tab2, tab3 = st.tabs(["Natural Language Query", "Form Input", "Spreadsheet Upload"])
    
    # Initialize variables
    source = None
    destination = None
    protocol = "TCP"
    port = "0"
    is_live = False
    submitted = False
    
    # Tab 1: Natural Language Query
    with tab1:
        with st.form("natural_language_form"):
            # Create a text area for natural language query
            # Users can type queries like "Find path from 10.210.1.10 to 2.2.2.2 using TCP port 80"
            query_text = st.text_area(
                "Enter your network path query",
                placeholder="Example: Find path from 10.210.1.10 to 2.2.2.2 using TCP port 80\nOr: Query path from 10.10.3.253 to 172.24.32.225 UDP port 53 with live data",
                help="Enter your query in natural language. Include source IP, destination IP, protocol (TCP/UDP), port (optional, defaults to 0), and optionally 'live data' or 'use live data'",
                height=100
            )
            
            # Create a submit button for the natural language form
            submitted_nl = st.form_submit_button("Query", use_container_width=True)
            
            if submitted_nl:
                submitted = True
                # Parse natural language query to extract parameters
                import re
                
                # Convert query to lowercase for easier parsing
                query_lower = query_text.lower() if query_text else ""
                
                # Check for live data keywords
                if any(keyword in query_lower for keyword in ['live data', 'use live', 'with live', 'live access']):
                    is_live = True
                
                # Extract IP addresses using regex
                # Pattern matches IPv4 addresses (e.g., 10.210.1.10, 192.168.1.1)
                ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
                ip_addresses = re.findall(ip_pattern, query_text) if query_text else []
                
                # Extract port number (optional - defaults to 0 if not specified)
                # Look for "port" followed by a number, or port after colon
                port_patterns = [
                    r'port\s+(\d{1,5})',  # "port 80" or "port 443"
                    r':(\d{1,5})(?:\s|$)',  # ":80" or ":443" (not part of IP)
                ]
                
                port_found = False
                for pattern in port_patterns:
                    port_match = re.search(pattern, query_text) if query_text else None
                    if port_match:
                        extracted_port = port_match.group(1)
                        # Validate port is in valid range (0-65535)
                        if extracted_port and 0 <= int(extracted_port) <= 65535:
                            port = extracted_port
                            port_found = True
                            break
                
                # Extract protocol (TCP or UDP)
                if 'udp' in query_lower:
                    protocol = "UDP"
                elif 'tcp' in query_lower:
                    protocol = "TCP"
                
                # Extract source and destination IPs
                # Look for keywords like "from", "to", "source", "destination"
                if len(ip_addresses) >= 2:
                    # Find positions of keywords
                    from_pos = query_lower.find('from')
                    to_pos = query_lower.find('to')
                    source_pos = query_lower.find('source')
                    dest_pos = query_lower.find('destination')
                    
                    # Determine which IP is source and which is destination
                    if from_pos != -1 and to_pos != -1:
                        # "from X to Y" pattern
                        from_ip_pos = query_text.lower().find(ip_addresses[0])
                        to_ip_pos = query_text.lower().find(ip_addresses[1])
                        if from_pos < from_ip_pos < to_pos < to_ip_pos:
                            source = ip_addresses[0]
                            destination = ip_addresses[1]
                        else:
                            source = ip_addresses[1]
                            destination = ip_addresses[0]
                    elif source_pos != -1 or dest_pos != -1:
                        # "source X destination Y" pattern
                        source = ip_addresses[0]
                        destination = ip_addresses[1]
                    else:
                        # Default: first IP is source, second is destination
                        source = ip_addresses[0]
                        destination = ip_addresses[1]
                elif len(ip_addresses) == 1:
                    # Only one IP found - assume it's source and ask for destination
                    source = ip_addresses[0]
                    st.error("Please provide both source and destination IP addresses in your query")
                    return
    
    # Tab 2: Form Input
    with tab2:
        # Create a form container named "network_query_form"
        # Forms allow grouping inputs and submitting them together
        with st.form("network_query_form"):
            # Create two columns side by side for better layout organization
            # col1 and col2 each take 50% of the available width
            col1, col2 = st.columns(2)
            
            # Place input fields in the first column (left side)
            with col1:
                # Create a text input field for source IP/hostname
                # "*" indicates required field, placeholder shows example format
                source_form = st.text_input("Source IP *", placeholder="e.g., 10.10.3.253")
                
                # Create a dropdown selectbox for protocol selection
                # Options are TCP and UDP, default selection is TCP (index=0)
                protocol_form = st.selectbox(
                    "Protocol *",  # Label with asterisk indicating required field
                    ["TCP", "UDP"],  # Available protocol options
                    index=0  # Default to first option (TCP)
                )
            
            # Place input fields in the second column (right side)
            with col2:
                # Create a text input field for destination IP/hostname
                # "*" indicates required field, placeholder shows example format
                destination_form = st.text_input("Destination IP *", placeholder="e.g., 172.24.32.225")
                
                # Create a text input field for port number
                # Port is optional - defaults to 0 if not provided
                port_form = st.text_input("Port", placeholder="e.g., 80, 443, 22 (default: 0)", value="0")
                
                # Create a checkbox for live data access
                # 0 = Baseline data, 1 = Live access
                is_live_form = st.checkbox("Use Live Data", value=False, help="Check to use live access data instead of baseline")
            
            # Create a submit button for the form
            # use_container_width=True makes the button span the full width of its container
            submitted_form = st.form_submit_button("Query", use_container_width=True)
            
            if submitted_form:
                submitted = True
                # Use form values
                source = source_form
                destination = destination_form
                protocol = protocol_form
                port = port_form.strip() if port_form and port_form.strip() else "0"
                is_live = is_live_form
    
    # Tab 3: Spreadsheet Upload
    with tab3:
        st.markdown("### Upload Spreadsheet")
        st.info("Upload a CSV or Excel file with columns: Source IP, Destination IP, Protocol, Port (optional), Use Live Data (optional)")
        
        # Create a file uploader for spreadsheet files
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file. Expected columns: Source IP (or source_ip, source), Destination IP (or destination_ip, destination), Protocol (TCP/UDP), Port (optional, defaults to 0), Use Live Data (optional, Yes/No, True/False, 1/0)"
        )
        
        if uploaded_file is not None:
            # Display file details
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Read the spreadsheet based on file type
            try:
                if uploaded_file.name.endswith('.csv'):
                    # Read CSV file
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    # Read Excel file
                    # First, read without headers to inspect the structure
                    df_temp = pd.read_excel(uploaded_file, header=None)
                    
                    # Check if row 0 has headers (text like "Source IP", "Destination", etc.)
                    row0_values = df_temp.iloc[0].astype(str).str.lower().tolist()
                    has_header_keywords = any(
                        any(keyword in str(val) for keyword in ['source', 'destination', 'dest', 'protocol', 'port', 'ip'])
                        for val in row0_values
                    )
                    
                    # Check if row 1 has headers (in case row 0 is empty or has different content)
                    if len(df_temp) > 1:
                        row1_values = df_temp.iloc[1].astype(str).str.lower().tolist()
                        has_header_keywords_row1 = any(
                            any(keyword in str(val) for keyword in ['source', 'destination', 'dest', 'protocol', 'port', 'ip'])
                            for val in row1_values
                        )
                    else:
                        has_header_keywords_row1 = False
                    
                    # Determine which row to use as header
                    if has_header_keywords:
                        # Use row 0 as header
                        df = pd.read_excel(uploaded_file, header=0)
                    elif has_header_keywords_row1:
                        # Use row 1 as header
                        df = pd.read_excel(uploaded_file, header=1)
                    else:
                        # Default: use row 0 as header
                        df = pd.read_excel(uploaded_file, header=0)
                else:
                    st.error("Unsupported file type. Please upload a CSV or Excel file.")
                    df = None
                
                if df is not None and not df.empty:
                    # Display the uploaded data preview
                    st.markdown("### Preview of Uploaded Data")
                    st.dataframe(df, use_container_width=True)
                    
                    # Normalize column names (case-insensitive, handle variations)
                    df.columns = df.columns.str.strip().str.lower()
                    
                    # Map common column name variations to standard names
                    # This includes handling typos and common misspellings
                    column_mapping = {
                        'source ip': 'source',
                        'source_ip': 'source',
                        'sourceip': 'source',
                        'src': 'source',
                        'destination ip': 'destination',
                        'destination_ip': 'destination',
                        'destinationip': 'destination',
                        'dest': 'destination',
                        'desitnation ip': 'destination',  # Handle typo: "Desitnation"
                        'desitnation_ip': 'destination',
                        'desitnationip': 'destination',
                        'destnation ip': 'destination',  # Handle typo: "Destnation"
                        'destnation_ip': 'destination',
                        'destnationip': 'destination',
                        'protocol': 'protocol',
                        'port': 'port',
                        'use live data': 'is_live',
                        'use_live_data': 'is_live',
                        'uselivedata': 'is_live',
                        'live': 'is_live',
                        'live data': 'is_live'
                    }
                    
                    # Rename columns based on mapping
                    df.rename(columns=column_mapping, inplace=True)
                    
                    # Additional fuzzy matching for columns that might have typos
                    # Check if we still have missing required columns and try fuzzy matching
                    required_columns = ['source', 'destination', 'protocol']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        # Try to find columns that might match with typos
                        available_columns = df.columns.tolist()
                        
                        # Fuzzy match for destination (common typo: "Desitnation")
                        if 'destination' in missing_columns:
                            for col in available_columns:
                                # Check if column name contains key parts of "destination"
                                col_lower = col.lower()
                                if 'dest' in col_lower and ('ip' in col_lower or 'address' in col_lower or len(col_lower) > 8):
                                    df.rename(columns={col: 'destination'}, inplace=True)
                                    if 'destination' in missing_columns:
                                        missing_columns.remove('destination')
                                    break
                        
                        # Fuzzy match for source
                        if 'source' in missing_columns:
                            for col in available_columns:
                                col_lower = col.lower()
                                if 'src' in col_lower or ('source' in col_lower and 'ip' in col_lower):
                                    df.rename(columns={col: 'source'}, inplace=True)
                                    if 'source' in missing_columns:
                                        missing_columns.remove('source')
                                    break
                        
                        # Fuzzy match for protocol
                        if 'protocol' in missing_columns:
                            for col in available_columns:
                                col_lower = col.lower()
                                if 'protocol' in col_lower or 'proto' in col_lower:
                                    df.rename(columns={col: 'protocol'}, inplace=True)
                                    if 'protocol' in missing_columns:
                                        missing_columns.remove('protocol')
                                    break
                    
                    # Check for required columns
                    required_columns = ['source', 'destination', 'protocol']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                        st.info("Required columns: Source IP, Destination IP, Protocol")
                    else:
                        # Process button with custom green background color
                        st.markdown("""
                            <style>
                            div[data-testid="stButton"] > button[kind="primary"] {
                                background-color: #4CAF50 !important;
                                color: white !important;
                                border: none !important;
                                font-weight: bold !important;
                            }
                            div[data-testid="stButton"] > button[kind="primary"]:hover {
                                background-color: #45a049 !important;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        if st.button("Process All Rows", use_container_width=True, type="primary"):
                            submitted = True
                            # Store the dataframe in session state for processing
                            st.session_state['spreadsheet_df'] = df
                            st.session_state['process_spreadsheet'] = True
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.info("Please ensure the file is a valid CSV or Excel file with the correct format.")
    
    # Check if spreadsheet processing is requested
    if st.session_state.get('process_spreadsheet', False) and 'spreadsheet_df' in st.session_state:
        df = st.session_state['spreadsheet_df']
        st.session_state['process_spreadsheet'] = False  # Reset flag
        
        # Process each row
        st.markdown("### Processing Spreadsheet Queries")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_list = []
        
        for idx, row in df.iterrows():
            # Update progress
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Processing row {idx + 1} of {len(df)}: {row.get('source', 'N/A')} → {row.get('destination', 'N/A')}")
            
            # Extract values from row
            source = str(row.get('source', '')).strip() if pd.notna(row.get('source')) else None
            destination = str(row.get('destination', '')).strip() if pd.notna(row.get('destination')) else None
            protocol = str(row.get('protocol', 'TCP')).strip().upper() if pd.notna(row.get('protocol')) else 'TCP'
            port = str(row.get('port', '0')).strip() if pd.notna(row.get('port')) else '0'
            
            # Handle is_live column (can be Yes/No, True/False, 1/0, or boolean)
            is_live_val = row.get('is_live', False)
            if pd.isna(is_live_val):
                is_live = False
            elif isinstance(is_live_val, bool):
                is_live = is_live_val
            elif isinstance(is_live_val, str):
                is_live_str = is_live_val.strip().lower()
                is_live = is_live_str in ['yes', 'true', '1', 'y']
            else:
                is_live = bool(is_live_val)
            
            # Validate row data
            if not source or not destination:
                results_list.append({
                    'row': idx + 1,
                    'source': source,
                    'destination': destination,
                    'status': 'Error',
                    'message': 'Missing source or destination IP'
                })
                continue
            
            # Set default port to "0" if empty
            if not port or port == '':
                port = "0"
            
            # Convert is_live to integer (0 or 1) for API
            is_live_value = 1 if is_live else 0
            
            # Execute query for this row
            try:
                async def execute_query():
                    """Execute the network path query asynchronously for a single row"""
                    server_params = get_server_params()
                    async with stdio_client(server_params) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            tool_arguments = {
                                "source": source,
                                "destination": destination,
                                "protocol": protocol,
                                "port": port,
                                "is_live": is_live_value
                            }
                            tool_result = await session.call_tool(
                                "query_network_path",
                                arguments=tool_arguments
                            )
                            if tool_result and tool_result.content:
                                import json
                                result_text = tool_result.content[0].text
                                try:
                                    return json.loads(result_text)
                                except json.JSONDecodeError:
                                    return {"result": result_text}
                            else:
                                return None
                
                result = asyncio.run(execute_query())
                
                # Store result
                results_list.append({
                    'row': idx + 1,
                    'source': source,
                    'destination': destination,
                    'protocol': protocol,
                    'port': port,
                    'is_live': is_live,
                    'result': result
                })
                
            except Exception as e:
                results_list.append({
                    'row': idx + 1,
                    'source': source,
                    'destination': destination,
                    'status': 'Error',
                    'message': str(e)
                })
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.markdown("### Results")
        
        # Create a summary table
        summary_data = []
        for res in results_list:
            if 'result' in res and res['result']:
                if 'error' in res['result']:
                    query_status = 'Error'
                    message = res['result'].get('error', 'Unknown error')
                    # Extract path result (failure reason) from path_hops or path_failure_reason
                    path_result = None
                    if 'path_failure_reason' in res['result']:
                        path_result = f"Reason: {res['result']['path_failure_reason']}"
                    elif 'path_hops' in res['result']:
                        # Look for failure reasons in path hops
                        for hop in res['result']['path_hops']:
                            if hop.get('failure_reason'):
                                path_result = f"Reason: {hop.get('failure_reason')}"
                                break
                else:
                    query_status = 'Success'
                    message = res['result'].get('statusDescription', 'Query completed')
                    # Extract path result from successful queries
                    path_result = None
                    if 'path_failure_reason' in res['result']:
                        path_result = f"Reason: {res['result']['path_failure_reason']}"
                    elif 'path_hops' in res['result']:
                        # Check if any hop has a failure reason
                        for hop in res['result']['path_hops']:
                            if hop.get('failure_reason'):
                                path_result = f"Reason: {hop.get('failure_reason')}"
                                break
                        # If no failure reason, path is successful
                        if path_result is None:
                            path_result = "Path calculation successful"
            else:
                query_status = res.get('status', 'Unknown')
                message = res.get('message', 'No result')
                path_result = None
            
            summary_data.append({
                'Row': res['row'],
                'Source': res.get('source', 'N/A'),
                'Destination': res.get('destination', 'N/A'),
                'Protocol': res.get('protocol', 'N/A'),
                'Port': res.get('port', 'N/A'),
                'Path result': path_result if path_result else message
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Display detailed results in expandable sections
        for res in results_list:
            with st.expander(f"Row {res['row']}: {res.get('source', 'N/A')} → {res.get('destination', 'N/A')}"):
                if 'result' in res and res['result']:
                    display_result(res['result'])
                else:
                    st.error(f"Error: {res.get('message', 'Unknown error')}")
        
        # Clear session state
        if 'spreadsheet_df' in st.session_state:
            del st.session_state['spreadsheet_df']
    
    # Check if either form was submitted (button clicked)
    if submitted:
        # Validate that required fields are filled
        # Check if source or destination are empty strings or None
        if not source or not destination:
            # Display an error message if any required field is missing
            st.error("Please provide source and destination IP addresses (fields marked with * are required)")
            # Exit the function early if validation fails
            return
        
        # Set default port to "0" if not provided or empty
        if not port or port.strip() == "":
            port = "0"
        else:
            port = port.strip()
        
        # Convert is_live checkbox value to integer (0 or 1) for API
        is_live_value = 1 if is_live else 0
        
        # Display a spinner animation while the query is being processed
        # This provides visual feedback to the user that work is in progress
        with st.spinner('Querying network...'):
            # Wrap the query execution in a try-except block to handle errors gracefully
            try:
                # Define an inner async function to execute the network query
                # This function handles the asynchronous MCP client communication
                async def execute_query():
                    """
                    Execute the network path query asynchronously.
                    
                    This inner function:
                    1. Creates MCP client session with proper context managers
                    2. Calls the query_network_path tool on the server
                    3. Extracts and parses the result
                    4. Returns the parsed result or None
                    
                    The context managers ensure proper cleanup of resources.
                    """
                    # Get server parameters for stdio communication
                    server_params = get_server_params()
                    
                    # Use stdio_client helper to create read/write streams for stdio transport
                    # stdio_client returns an async context manager that provides (read_stream, write_stream)
                    # We use async with to properly manage the context lifecycle
                    # This ensures the subprocess is properly cleaned up when done
                    async with stdio_client(server_params) as (read_stream, write_stream):
                        # Create a new ClientSession instance with the read and write streams
                        # ClientSession requires both read_stream and write_stream for communication
                        # We also use async with for ClientSession to properly manage its lifecycle
                        async with ClientSession(read_stream, write_stream) as session:
                            # Initialize the MCP protocol handshake with the server
                            # This exchanges capabilities and sets up the communication protocol
                            await session.initialize()
                            
                            # Call the MCP tool named "query_network_path" on the server
                            # This sends a request to the MCP server to execute the tool
                            # await is needed because call_tool() is an async method
                            # Build arguments dictionary with required parameters
                            # Gateway is now automatically resolved by the API (Step 1)
                            tool_arguments = {
                                "source": source,  # Source IP/hostname from form input
                                "destination": destination,  # Destination IP/hostname from form input
                                "protocol": protocol,  # Selected protocol (TCP or UDP)
                                "port": port,  # Port number from form input
                                "is_live": is_live_value  # Convert checkbox boolean to integer (0 or 1)
                            }
                            
                            # Call the tool with all arguments
                            tool_result = await session.call_tool(
                                "query_network_path",  # Name of the tool to call (defined in mcp_server.py)
                                arguments=tool_arguments  # Dictionary of arguments to pass to the tool
                            )
                            
                            # Extract the result from the tool response
                            # Check if tool_result exists and has content
                            if tool_result and tool_result.content:
                                # Import json module for parsing JSON strings
                                import json
                                # Extract the text content from the first content item in the response
                                # MCP responses contain content as a list, we take the first item's text
                                result_text = tool_result.content[0].text
                                # Try to parse the text as JSON
                                try:
                                    # Parse the JSON string into a Python dictionary
                                    return json.loads(result_text)
                                except json.JSONDecodeError:
                                    # If parsing fails, return the text as a dictionary with a "result" key
                                    # This handles cases where the server returns plain text instead of JSON
                                    return {"result": result_text}
                            else:
                                # Return None if no content was returned from the tool
                                return None
                
                # Run the async execute_query function using asyncio.run()
                # This executes the async function and waits for its completion
                # asyncio.run() is needed because main() is a synchronous function
                result = asyncio.run(execute_query())
                
                # Display the results to the user using the helper function
                if result:
                    display_result(result)
                else:
                    # Display a warning message if no results were returned
                    st.warning("No results returned")
                    
            # Catch any exceptions that occur during query execution
            except Exception as e:
                # Display the error message to the user
                st.error(f"An error occurred: {e}")
                # Import traceback module for detailed error information
                import traceback
                # Display the full traceback for debugging purposes
                # This shows the complete call stack when an error occurs
                st.error(traceback.format_exc())

# Check if this script is being run directly (not imported as a module)
# __name__ will be "__main__" when the script is executed directly
# This allows the script to be both runnable and importable
if __name__ == "__main__":
    # Call the main function to start the Streamlit application
    # This will launch the web interface when the script is run
    main()
