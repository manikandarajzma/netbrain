# Device Type Icons

This directory should contain icon image files for different device types used in the network path visualization.

## Icon File Naming Convention

Place icon files in this directory with the following names:

- `paloalto_firewall.png` - For Palo Alto Networks firewalls
- `arista_switch.png` - For Arista switches
- `cisco_device.png` - For Cisco devices
- `juniper_device.png` - For Juniper devices
- `generic_switch.png` - For generic switches
- `generic_router.png` - For generic routers
- `generic_firewall.png` - For generic firewalls
- `default_device.png` - Default icon for unknown device types

## Icon Specifications

- **Format**: PNG (recommended) or other image formats supported by PIL
- **Size**: 64x64 pixels (will be resized automatically)
- **Background**: Transparent or solid color
- **Style**: Should match NetBrain UI style if possible

## Fallback Behavior

If icon files are not found, the system will automatically generate programmatic icons:
- **Palo Alto Firewalls**: Red square with "FW" text
- **Arista Switches**: Blue square with "A" text
- **Cisco Devices**: Light blue square with "C" text
- **Switches**: Blue square with "SW" text
- **Routers**: Green square with "R" text
- **Default**: Gray square with "?" text

## Adding Custom Icons

1. Create or download icon images matching the naming convention above
2. Place them in this `icons/` directory
3. Restart the Streamlit application
4. The icons will automatically be used in network path visualizations
