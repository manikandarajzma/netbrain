"""
Standalone test script for Panorama device group extraction.
Tests device group querying for firewalls.
"""

import asyncio
import sys
import os

# Add the netbrain directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path is set
import panoramaauth


async def test_device_groups():
    """
    Test Panorama device group extraction with hardcoded firewall names.
    """
    print("=" * 60)
    print("Testing Panorama Device Group Extraction")
    print("=" * 60)
    
    # Hardcoded firewall names for testing
    firewall_names = ["roundrock-dc-fw1", "leander-dc-fw1"]
    
    print(f"\nFirewalls to test: {firewall_names}")
    print("\n" + "-" * 60)
    
    try:
        # Test device group extraction
        print("Calling get_device_groups_for_firewalls...")
        device_groups = await panoramaauth.get_device_groups_for_firewalls(
            firewall_names=firewall_names
        )
        
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(f"Device groups returned: {device_groups}")
        
        # Display results
        for firewall_name, device_group in device_groups.items():
            if device_group:
                print(f"  [OK] {firewall_name} -> Device Group: {device_group}")
            else:
                print(f"  [FAIL] {firewall_name} -> Device Group: None (not found)")
        
        print("\n" + "=" * 60)
        
        # Check if all firewalls got device groups
        all_found = all(group is not None for group in device_groups.values())
        if all_found:
            print("[SUCCESS] All firewalls have device groups!")
        else:
            print("[WARNING] Some firewalls don't have device groups")
        
        return device_groups
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        print("\nTraceback:")
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    print("Starting Panorama device group extraction test...\n")
    result = asyncio.run(test_device_groups())
    
    if result:
        print(f"\nTest completed. Device Groups: {result}")
    else:
        print("\nTest failed.")
        sys.exit(1)
