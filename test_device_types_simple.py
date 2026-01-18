"""
Simple test for the getAllDisplayDeviceTypes endpoint.
"""

import requests
import json
import urllib3
import netbrainauth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

token = netbrainauth.get_auth_token()
if not token:
    print("ERROR: Could not get authentication token")
    exit(1)

print(f"Token (first 50 chars): {token[:50]}...")
print(f"Token length: {len(token)}")
print(f"Token type: {type(token)}")

nb_url = netbrainauth.NETBRAIN_URL
# Try Token header first (standard NetBrain API authentication)
# If that fails, we'll try Bearer token
headers_token = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Token': token
}
headers_bearer = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {token}'
}

endpoint = "/ServicesAPI/SystemModel/getAllDisplayDeviceTypes"
full_url = nb_url + endpoint

print(f"Testing: {full_url}")
print("="*80)

try:
    # Try Token header first
    print("Trying with Token header...")
    response = requests.get(full_url, headers=headers_token, verify=False, timeout=10)
    print(f"Status Code (Token): {response.status_code}")
    
    # If Token header fails with 401, try Bearer
    if response.status_code == 401:
        print("\nToken header failed, trying Bearer token...")
        response = requests.get(full_url, headers=headers_bearer, verify=False, timeout=10)
        print(f"Status Code (Bearer): {response.status_code}")
    
    print(f"Response Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nResponse Type: {type(data)}")
        print(f"Response (full):\n{json.dumps(data, indent=2)}")
        
        # Try to extract device type mappings
        device_type_map = {}
        
        # Handle different response structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try common keys
            items = data.get('result') or data.get('data') or data.get('deviceTypes') or data.get('items') or []
            if not isinstance(items, list):
                items = [items] if items else []
        
        print(f"\nFound {len(items)} items")
        
        for item in items:
            if isinstance(item, dict):
                # Try different field names for ID and name
                dt_id = None
                dt_name = None
                
                # Try ID fields
                for id_field in ['id', 'ID', 'deviceTypeId', 'deviceTypeID', 'typeId', 'typeID', 'devType', 'devTypeId', 'deviceType']:
                    if id_field in item:
                        try:
                            dt_id = int(item[id_field])
                            break
                        except (ValueError, TypeError):
                            pass
                
                # Try name fields
                for name_field in ['deviceType', 'DeviceType', 'name', 'Name', 'typeName', 'TypeName', 'description', 'Description', 'displayName', 'DisplayName']:
                    if name_field in item:
                        dt_name = str(item[name_field])
                        break
                
                if dt_id and dt_name:
                    device_type_map[dt_id] = dt_name
                    print(f"  Mapped: {dt_id} -> {dt_name}")
        
        if device_type_map:
            print(f"\n[SUCCESS] Found {len(device_type_map)} device type mappings!")
            print("\nSample mappings:")
            for dt_id, dt_name in list(device_type_map.items())[:20]:
                print(f"  {dt_id}: {dt_name}")
            
            # Test with known device types
            test_types = [1036, 2013, 2130, 0]
            print("\nTesting with known device types:")
            for dt_id in test_types:
                dt_name = device_type_map.get(dt_id, f"NOT FOUND (Device Type {dt_id})")
                print(f"  {dt_id} -> {dt_name}")
        else:
            print("\n[WARNING] Could not extract device type mappings")
            print("Item structure:")
            if items:
                print(json.dumps(items[0], indent=2))
    else:
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
