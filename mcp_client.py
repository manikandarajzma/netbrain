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

# Configure Streamlit page settings:
# - page_title: Sets the browser tab title to "NetBrain Network Query"
# - page_icon: Sets the browser tab icon to a globe emoji (🌐)
# - layout: Sets the page layout to "centered" for better visual presentation
st.set_page_config(
    page_title="NetBrain Network Query",
    page_icon="🌐",
    layout="centered"
)

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
    
    # Create tabs to switch between natural language and form input
    tab1, tab2 = st.tabs(["Natural Language Query", "Form Input"])
    
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
                
                # Display the results to the user
                # Check if result exists (is not None and not empty)
                if result:
                    # Check if result is a dictionary
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
                    # Check if result is a string
                    elif isinstance(result, str):
                        # Display success message
                        st.success("Query completed")
                        # Display the string result as plain text
                        st.text(result)
                    else:
                        # For any other result type, display as JSON
                        st.json(result)
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
