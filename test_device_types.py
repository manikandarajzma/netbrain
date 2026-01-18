"""
Test script to query NetBrain API for device type mappings.
This script will try various API endpoints to find the correct one for device types.
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

print(f"Token obtained: {token[:20]}...")

# NetBrain URL
nb_url = netbrainauth.NETBRAIN_URL
print(f"NetBrain URL: {nb_url}")

# Headers
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Token': token
}

# List of potential API endpoints to try
endpoints = [
    "/ServicesAPI/SystemModel/getAllDisplayDeviceTypes",  # The correct endpoint!
    "/ServicesAPI/API/V1/CMDB/DeviceType",
    "/ServicesAPI/API/V1/CMDB/DeviceTypes",
    "/ServicesAPI/API/V1/Admin/DeviceType",
    "/ServicesAPI/API/V1/Admin/DeviceTypes",
]

print("\n" + "="*80)
print("Testing NetBrain API endpoints for device types...")
print("="*80 + "\n")

device_type_map = {}

for endpoint in endpoints:
    full_url = nb_url + endpoint
    print(f"Testing: {endpoint}")
    print(f"Full URL: {full_url}")
    
    # Try GET first, then POST if GET fails
    try:
        response = requests.get(full_url, headers=headers, verify=False, timeout=10)
        if response.status_code == 404:
            # Try POST for some endpoints
            if "DeviceType" in endpoint or "DeviceTypes" in endpoint:
                print("  Trying POST method...")
                response = requests.post(full_url, headers=headers, json={}, verify=False, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Response Type: {type(data)}")
                print(f"Response Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                print(f"Response (first 1000 chars): {json.dumps(data, indent=2)[:1000]}")
                
                # Try to extract device types
                device_types = []
                if isinstance(data, dict):
                    # Try different possible keys
                    for key in ['deviceTypes', 'deviceType', 'result', 'data', 'items', 'list']:
                        if key in data:
                            value = data[key]
                            if isinstance(value, list):
                                device_types = value
                                break
                            elif isinstance(value, dict):
                                # Nested structure
                                for nested_key in ['deviceTypes', 'deviceType', 'items', 'list']:
                                    if nested_key in value and isinstance(value[nested_key], list):
                                        device_types = value[nested_key]
                                        break
                                if device_types:
                                    break
                elif isinstance(data, list):
                    device_types = data
                
                # Check if this is the devices endpoint - it has a different structure
                if "devices" in data and isinstance(data["devices"], list):
                    devices = data["devices"]
                    print(f"\nFound {len(devices)} devices!")
                    print("\nFirst few devices:")
                    for i, dev in enumerate(devices[:5]):
                        print(f"  {i+1}. {json.dumps(dev, indent=4)}")
                    
                    # Check if devices have device type codes
                    print("\nChecking for device type codes in device objects...")
                    type_code_map = {}
                    for dev in devices:
                        if isinstance(dev, dict):
                            # Look for device type code fields
                            type_code = None
                            type_name = None
                            
                            # Try various code field names
                            for code_field in ['devType', 'deviceType', 'typeId', 'typeID', 'deviceTypeId', 'deviceTypeID', 'subTypeId', 'subTypeID']:
                                if code_field in dev:
                                    try:
                                        type_code = int(dev[code_field])
                                        break
                                    except (ValueError, TypeError):
                                        pass
                            
                            # Get type name
                            type_name = dev.get('subTypeName') or dev.get('deviceType') or dev.get('typeName') or dev.get('name')
                            
                            if type_code and type_name:
                                type_code_map[type_code] = type_name
                                print(f"  Found: {type_code} -> {type_name}")
                    
                    if type_code_map:
                        device_type_map.update(type_code_map)
                        print(f"\n[SUCCESS] Found {len(type_code_map)} device type mappings from devices!")
                        break
                    else:
                        print("[WARNING] Devices don't contain device type codes")
                
                elif device_types:
                    print(f"\nFound {len(device_types)} device types!")
                    print("\nFirst few device types:")
                    for i, dt in enumerate(device_types[:5]):
                        print(f"  {i+1}. {json.dumps(dt, indent=4)}")
                    
                    # Try to extract mappings
                    print("\nExtracting device type mappings...")
                    for dt in device_types:
                        if isinstance(dt, dict):
                            # Try different field names
                            dt_id = None
                            dt_name = None
                            
                            # Try various ID field names
                            for id_field in ['id', 'ID', 'deviceTypeId', 'deviceTypeID', 'typeId', 'typeID', 'devType', 'devTypeId']:
                                if id_field in dt:
                                    try:
                                        dt_id = int(dt[id_field])
                                        break
                                    except (ValueError, TypeError):
                                        pass
                            
                            # Try various name field names
                            for name_field in ['deviceType', 'DeviceType', 'name', 'Name', 'typeName', 'TypeName', 'description', 'Description']:
                                if name_field in dt:
                                    dt_name = str(dt[name_field])
                                    break
                            
                            if dt_id and dt_name:
                                device_type_map[dt_id] = dt_name
                                print(f"  Mapped: {dt_id} -> {dt_name}")
                    
                    if device_type_map:
                        print(f"\n[SUCCESS] Found {len(device_type_map)} device type mappings")
                        print("\nSample mappings:")
                        for dt_id, dt_name in list(device_type_map.items())[:10]:
                            print(f"  {dt_id}: {dt_name}")
                        break  # Found working endpoint, exit loop
                    else:
                        print("[WARNING] Found device types but couldn't extract ID/name mappings")
                else:
                    print("[WARNING] Response doesn't contain device types list")
                
            except json.JSONDecodeError as e:
                print(f"[WARNING] Response is not valid JSON: {e}")
                print(f"Response text: {response.text[:500]}")
        elif response.status_code == 404:
            print("[FAIL] Endpoint not found (404)")
        elif response.status_code == 401:
            print("[FAIL] Unauthorized (401) - Check token")
        elif response.status_code == 403:
            print("[FAIL] Forbidden (403) - Check permissions")
        else:
            print(f"[FAIL] Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
    
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Request failed: {e}")
    
    print("\n" + "-"*80 + "\n")

if device_type_map:
    print("="*80)
    print(f"FINAL RESULT: Found {len(device_type_map)} device type mappings")
    print("="*80)
    print("\nAll mappings:")
    for dt_id in sorted(device_type_map.keys()):
        print(f"  {dt_id}: {device_type_map[dt_id]}")
    
    # Test with known device types from the graph
    test_types = [1036, 2013, 2130, 0]
    print("\n" + "="*80)
    print("Testing with known device types from graph:")
    print("="*80)
    for dt_id in test_types:
        dt_name = device_type_map.get(dt_id, f"NOT FOUND (Device Type {dt_id})")
        print(f"  {dt_id} -> {dt_name}")
else:
    print("="*80)
    print("[FAIL] Could not find device type mappings from any endpoint")
    print("="*80)
    print("\nTried endpoints:")
    for endpoint in endpoints:
        print(f"  - {endpoint}")
