"""
NetBrain API Authentication Module
Handles authentication with NetBrain API using Session API.

This module provides:
- Username/password authentication via NetBrain Session API
- Token caching to avoid unnecessary API calls
- Support for external users (LDAP/AD/TACACS) via authentication_id
- SSL certificate verification bypass for self-signed certificates

Reference: https://github.com/NetBrainAPI/NetBrain-REST-API-R11/blob/main/REST%20APIs%20Documentation/Authentication%20and%20Authorization/Login%20API.md
"""

# Import requests library for making HTTP POST requests to the Session endpoint
import requests

# Import time module for timestamp operations (if needed for token expiry)
import time

# Import urllib3 for SSL warning suppression
import urllib3

# Import json module for JSON serialization
# Used to convert Python dictionaries to JSON strings for the request body
import json

# Import os module for accessing environment variables
# Used to read NetBrain configuration from environment
import os

# Import Optional type hint from typing module
# Used to indicate that a function can return None
from typing import Optional

# Disable SSL warnings from urllib3
# This suppresses warnings about unverified SSL certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# NetBrain API configuration
# Hardcoded configuration values for NetBrain API authentication
# These can be changed directly in the code or overridden via environment variables

# Load .env file if available
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

# NetBrain server URL
NETBRAIN_URL = os.getenv("NETBRAIN_URL", "http://localhost")

# NetBrain username for authentication
USERNAME = os.getenv("NETBRAIN_USERNAME", "")

# NetBrain password for authentication
PASSWORD = os.getenv("NETBRAIN_PASSWORD", "")

# Authentication ID for external users (LDAP/AD/TACACS)
AUTHENTICATION_ID = os.getenv("NETBRAIN_AUTH_ID") or None

# Cache for access token
# Module-level variables to store the cached token
# These persist across function calls within the same Python process

# _access_token: Stores the current NetBrain session token
# Optional[str] means it can be a string or None
# Underscore prefix indicates it's a private module-level variable
_access_token: Optional[str] = None

# Timestamp when the token was obtained (for TTL-based expiry)
_token_obtained_at: float = 0.0

# Token time-to-live in seconds (30 minutes)
# NetBrain tokens can expire server-side; this ensures we refresh proactively
TOKEN_TTL_SECONDS = 30 * 60


def get_auth_token() -> Optional[str]:
    """
    Get NetBrain API session token using username/password authentication.
    Returns cached token if available and not expired, otherwise requests a new one.

    This function implements token caching to minimize API calls:
    - Checks if a cached token exists and hasn't exceeded TOKEN_TTL_SECONDS
    - If available and fresh, returns the cached token immediately
    - If missing or expired, requests a new token from the Session API
    - Caches the new token for reuse

    Returns:
        Optional[str]: Session token string if successful, None if authentication fails
    """
    # Declare that we're modifying global variables
    global _access_token, _token_obtained_at

    # Check if we have a cached token that hasn't expired
    if _access_token and (time.time() - _token_obtained_at) < TOKEN_TTL_SECONDS:
        # Return the cached token immediately (no API call needed)
        return _access_token

    # Token is missing or expired - clear it and request a fresh one
    if _access_token:
        print(f"NetBrain token expired (age: {time.time() - _token_obtained_at:.0f}s, TTL: {TOKEN_TTL_SECONDS}s), refreshing...", file=__import__('sys').stderr, flush=True)
        _access_token = None
    
    # Validate that USERNAME and PASSWORD are configured
    if not USERNAME or not PASSWORD:
        print("Warning: NetBrain USERNAME or PASSWORD not configured")
        return None
    
    # Construct the NetBrain Session API endpoint URL
    full_url = f"{NETBRAIN_URL}/ServicesAPI/API/V1/Session"
    
    # Build the request body payload
    body = {
        "username": USERNAME,
        "password": PASSWORD
    }
    
    # Add authentication_id if provided (for external users)
    if AUTHENTICATION_ID:
        body["authentication_id"] = AUTHENTICATION_ID
    
    # Prepare HTTP headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Wrap API call in try-except to handle errors gracefully
    try:
        # Make HTTP POST request to NetBrain Session API endpoint
        response = requests.post(
            full_url,
            headers=headers,
            data=json.dumps(body),
            verify=False,
            timeout=10
        )
        
        # Check if the HTTP response status code is 200 (OK)
        if response.status_code == 200:
            # Parse the JSON response body
            js = response.json()
            
            # Check if the status code indicates success
            if js.get("statusCode") == 790200:
                # Extract the token from the response
                _access_token = js.get("token")
                _token_obtained_at = time.time()

                # Return the access token to the caller
                return _access_token
            else:
                # If status code is not 790200, authentication failed
                status_code = js.get("statusCode", "Unknown")
                status_description = js.get("statusDescription", "No description")
                print(f"NetBrain authentication failed: statusCode={status_code}, statusDescription={status_description}")
                return None
        else:
            # If HTTP status code is not 200, authentication failed
            print(f"Get token failed! - {response.text}")
            return None
            
    except Exception as e:
        # Print error message to console
        print(f"Error getting NetBrain auth token: {e}")
        return None


def clear_token_cache():
    """
    Clear the cached access token (useful for testing or forced re-authentication).
    
    This function resets the token cache, forcing the next call to get_auth_token()
    to request a fresh token from the API. Useful for:
    - Testing authentication flows
    - Forcing token refresh when needed
    - Handling cases where token might be invalidated server-side
    """
    # Declare that we're modifying global variables
    global _access_token, _token_obtained_at

    # Reset the cached token and timestamp
    _access_token = None
    _token_obtained_at = 0.0


# Test code (only runs when script is executed directly, not when imported)
# This allows testing the authentication without running the full application
if __name__ == "__main__":
    # Test authentication with current configuration
    print("=" * 60)
    print("Testing NetBrain Authentication")
    print("=" * 60)
    print(f"NetBrain URL: {NETBRAIN_URL}")
    print(f"Username: {USERNAME}")
    print("=" * 60)
    
    # Clear any cached tokens to force fresh authentication
    clear_token_cache()
    
    # Get authentication token (simple username/password authentication)
    token = get_auth_token()
    
    if token:
        print("=" * 60)
        print("[SUCCESS] Authentication successful!")
        print(f"Token: {token[:50]}...")
        print("=" * 60)
    else:
        print("=" * 60)
        print("[FAILED] Authentication failed!")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Verify your username and password are correct")
        print("2. Check that the NetBrain server is accessible")
        print("3. Verify the API endpoint URL is correct")
        print("4. Check if your account is locked or expired")
