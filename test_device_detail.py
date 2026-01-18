"""
Test script to query a specific device to see if it includes device type code.
"""

import requests
import json
import urllib3
import netbrainauth

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Get authentication token
token = netbrainauth.get_auth_token()
if not token:
    print("ERROR: Could not get authentication token")
    exit(1)

nb_url = netbrainauth.NETBRAIN_URL
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Token': token
}

# Test with a known device from the path
test_device = "roundrock-dc-fw1"

print(f"Querying device details for: {test_device}")
print("="*80)

# Try different endpoints for device details
endpoints = [
    f"/ServicesAPI/API/V1/CMDB/Device/{test_device}",
    f"/ServicesAPI/API/V1/CMDB/Devices/{test_device}",
    f"/ServicesAPI/API/V1/CMDB/Device?name={test_device}",
    f"/ServicesAPI/API/V1/CMDB/Devices?name={test_device}",
]

for endpoint in endpoints:
    full_url = nb_url + endpoint
    print(f"\nTesting: {endpoint}")
    print(f"Full URL: {full_url}")
    
    try:
        response = requests.get(full_url, headers=headers, verify=False, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response:\n{json.dumps(data, indent=2)}")
            
            # Look for device type code
            if isinstance(data, dict):
                print("\nSearching for device type code fields...")
                for key, value in data.items():
                    if 'type' in key.lower() or 'devtype' in key.lower():
                        print(f"  {key}: {value} (type: {type(value)})")
        else:
            print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")
