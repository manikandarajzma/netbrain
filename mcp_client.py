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
from mcp import ClientSession, StdioServerParameters

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

async def get_client():
    """
    Get or create MCP client session.
    
    This function manages the MCP client connection lifecycle:
    - Checks if a client session already exists in Streamlit's session state
    - If not, creates a new client session configured to connect to mcp_server.py via stdio
    - Initializes the client connection asynchronously
    - Returns the existing or newly created client session
    
    Returns:
        ClientSession: The MCP client session ready to make tool calls
    """
    # Check if 'mcp_client' key exists in Streamlit's session state dictionary
    # Session state persists across reruns, allowing us to reuse the connection
    if 'mcp_client' not in st.session_state:
        # Create server parameters for stdio communication:
        # - command: The command to run (python interpreter)
        # - args: Arguments to pass (the mcp_server.py script)
        # This configures the client to spawn mcp_server.py as a subprocess
        server_params = StdioServerParameters(
            command="python",  # Use Python interpreter to run the server
            args=["mcp_server.py"]  # Pass mcp_server.py as argument
        )
        # Create a new ClientSession instance with the server parameters
        # This session will communicate with the MCP server via standard input/output
        st.session_state.mcp_client = ClientSession(server_params)
        # Enter the async context manager (equivalent to __enter__ but async)
        # This establishes the connection to the server
        await st.session_state.mcp_client.__aenter__()
        # Initialize the MCP protocol handshake with the server
        # This exchanges capabilities and sets up the communication protocol
        await st.session_state.mcp_client.initialize()
    # Return the client session (either existing or newly created)
    return st.session_state.mcp_client

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
            source = st.text_input("Source *", placeholder="e.g., 192.168.1.1 or hostname")
            
            # Create a dropdown selectbox for protocol selection
            # Options are TCP and UDP, default selection is TCP (index=0)
            protocol = st.selectbox(
                "Protocol *",  # Label with asterisk indicating required field
                ["TCP", "UDP"],  # Available protocol options
                index=0  # Default to first option (TCP)
            )
        
        # Place input fields in the second column (right side)
        with col2:
            # Create a text input field for destination IP/hostname
            # "*" indicates required field, placeholder shows example format
            destination = st.text_input("Destination *", placeholder="e.g., 192.168.1.100 or hostname")
            
            # Create a text input field for port number
            # "*" indicates required field, placeholder shows example ports
            port = st.text_input("Port *", placeholder="e.g., 80, 443, 22")
        
        # Create a submit button for the form
        # use_container_width=True makes the button span the full width of its container
        submitted = st.form_submit_button("Query", use_container_width=True)
    
    # Check if the form was submitted (button clicked)
    if submitted:
        # Validate that all required fields are filled
        # Check if source, destination, or port are empty strings or None
        if not source or not destination or not port:
            # Display an error message if any required field is missing
            st.error("Please provide source, destination, and port (all fields marked with * are required)")
            # Exit the function early if validation fails
            return
        
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
                    1. Gets or creates the MCP client session
                    2. Calls the query_network_path tool on the server
                    3. Extracts and parses the result
                    4. Returns the parsed result or None
                    """
                    # Get or create the MCP client session (reuses existing if available)
                    # await is needed because get_client() is an async function
                    client = await get_client()
                    
                    # Call the MCP tool named "query_network_path" on the server
                    # This sends a request to the MCP server to execute the tool
                    # await is needed because call_tool() is an async method
                    tool_result = await client.call_tool(
                        "query_network_path",  # Name of the tool to call (defined in mcp_server.py)
                        arguments={  # Dictionary of arguments to pass to the tool
                            "source": source,  # Source IP/hostname from form input
                            "destination": destination,  # Destination IP/hostname from form input
                            "protocol": protocol,  # Selected protocol (TCP or UDP)
                            "port": port  # Port number from form input
                        }
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
                        else:
                            # Display success message in green using st.success()
                            st.success("Query completed successfully")
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
