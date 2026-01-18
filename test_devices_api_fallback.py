"""
Test script to verify the Devices API fallback for device type mapping.
This tests the approach used in mcp_server.py when direct device type endpoints fail.
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

nb_url = netbrainauth.NETBRAIN_URL
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Token': token
}

# Test the Devices API endpoint (fallback method)
devices_endpoint = f"{nb_url}/ServicesAPI/API/V1/CMDB/Devices"
params = {
    "version": 1,
    "skip": 0,
    "limit": 100  # Get first 100 devices
}

print("="*80)
print("Testing Devices API fallback for device type mapping")
print("="*80)
print(f"\nEndpoint: {devices_endpoint}")
print(f"Params: {params}\n")

try:
    response = requests.get(devices_endpoint, headers=headers, params=params, verify=False, timeout=10)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if data.get("statusCode") == 790200:
            devices = data.get("devices", [])
            total = data.get("totalCount", 0)
            
            print(f"\n[SUCCESS] Fetched {len(devices)} devices (total: {total})")
            
            # Extract device type mappings
            device_type_map = {}  # Numeric ID -> name (if available)
            device_name_to_type = {}  # Device name -> subTypeName (this is what we'll use)
            
            print("\nExtracting device information...")
            for dev in devices:
                dev_name = dev.get("name") or dev.get("hostname") or dev.get("mgmtIP")
                dev_type_name = dev.get("subTypeName") or dev.get("deviceTypeName") or dev.get("typeName")
                
                # Try to find numeric device type ID
                dev_type_id = dev.get("deviceType") or dev.get("devType") or dev.get("typeId")
                
                if dev_type_id and dev_type_name:
                    try:
                        device_type_map[int(dev_type_id)] = str(dev_type_name)
                    except (ValueError, TypeError):
                        pass
                
                # Build name-based cache (this is what we'll actually use)
                if dev_name and dev_type_name:
                    device_name_to_type[str(dev_name)] = str(dev_type_name)
                    print(f"  Device: '{dev_name}' -> Type: '{dev_type_name}'")
            
            # Show name-based cache first (this is what the code uses)
            print("\n" + "="*80)
            print("Device Name -> Type Mappings (Name-based cache - PRIMARY METHOD):")
            print("="*80)
            if device_name_to_type:
                print(f"\n[SUCCESS] Built name-based cache with {len(device_name_to_type)} entries!")
                print(f"{'Device Name':<30} {'Device Type':<50}")
                print("-" * 80)
                for dev_name, dev_type in sorted(device_name_to_type.items()):
                    print(f"{dev_name:<30} {dev_type:<50}")
                
                # Test with device names from the path
                print("\n" + "="*80)
                print("Testing with device names from path visualization:")
                print("="*80)
                test_device_names = ["hub", "roundrock-sw-1", "roundrock-dc-fw1", "leander-dc-fw1", "leander-sw-1", "10.0.0.254", "10.0.1.254"]
                for dev_name in test_device_names:
                    dev_type = device_name_to_type.get(dev_name, f"NOT FOUND (device: {dev_name})")
                    print(f"  '{dev_name}' -> '{dev_type}'")
            else:
                print("\n[WARNING] Could not build name-based cache")
            
            # Show numeric ID mappings if available
            if device_type_map:
                print("\n" + "="*80)
                print("Numeric ID -> Type Mappings (if available - FALLBACK METHOD):")
                print("="*80)
                print(f"\n[SUCCESS] Extracted {len(device_type_map)} unique numeric device type mappings!")
                print(f"{'Type ID':<15} {'Type Name':<50}")
                print("-" * 80)
                for dt_id, dt_name in sorted(device_type_map.items()):
                    print(f"{dt_id:<15} {dt_name:<50}")
                
                # Test with known device types
                print("\n" + "="*80)
                print("Testing with known device type codes from path hops:")
                print("="*80)
                test_types = [1036, 2013, 2130, 2001, 0]
                for dt_id in test_types:
                    dt_name = device_type_map.get(dt_id, f"NOT FOUND (Device Type {dt_id})")
                    print(f"  {dt_id} -> {dt_name}")
            else:
                print("\n[INFO] No numeric device type IDs found in device responses")
                print("This is expected - we'll use name-based lookup instead")
                print("\nSample device structure:")
                if devices:
                    print(json.dumps(devices[0], indent=2))
        else:
            print(f"\n[FAIL] API returned statusCode: {data.get('statusCode')}")
            print(f"Description: {data.get('statusDescription', 'No description')}")
    else:
        print(f"\n[FAIL] HTTP {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
except Exception as e:
    print(f"\n[ERROR] Exception: {e}")
    import traceback
    traceback.print_exc()
