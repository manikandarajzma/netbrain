"""
Main entry point for ai-netbrain project.

This is a simple placeholder module that can be used as the project's
main entry point. Currently, it just prints a greeting message.

The actual functionality is implemented in:
- app_fastapi.py: FastAPI web app and chat UI for network queries
- mcp_client.py: MCP client library (used by chat service)
- mcp_server.py: MCP server that handles network path queries
- netbrainauth.py: OAuth2 authentication module
"""

# Define the main function
# This function serves as the entry point for the application
def main():
    """
    Main function that prints a greeting message.
    
    This is a placeholder function. The actual application logic
    is in app_fastapi.py (FastAPI chat UI) and mcp_server.py (MCP server).
    
    To run the actual application:
    - For client: uv run python -m netbrain.app_fastapi
    - For server: python mcp_server.py
    """
    # Print a greeting message to the console
    # This confirms the module is being executed
    print("Hello from ai-netbrain!")


# Check if this script is being run directly (not imported as a module)
# __name__ will be "__main__" when the script is executed directly
# This allows the script to be both runnable and importable
if __name__ == "__main__":
    # Call the main function to execute the application
    # This will print the greeting message when the script is run
    main()
