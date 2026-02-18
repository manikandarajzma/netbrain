export function deviceTypeToIcon(deviceType) {
  if (!deviceType || typeof deviceType !== 'string') return null
  const t = deviceType.toLowerCase().trim()
  if (t.includes('palo alto') || (t.includes('firewall') && t.includes('palo'))) return '/icons/paloalto_firewall.png'
  if (t.includes('arista') || t.includes('switch')) return '/icons/arista_switch.png'
  if (t.includes('firewall')) return '/icons/paloalto_firewall.png'
  return null
}

export function deviceTypeToFallbackLabel(deviceType) {
  if (!deviceType || typeof deviceType !== 'string') return '?'
  const t = deviceType.toLowerCase().trim()
  if (t.includes('firewall')) return 'FW'
  if (t.includes('switch') || t.includes('arista')) return 'SW'
  if (t.includes('hub')) return 'HUB'
  return '?'
}

export function isFirewallType(deviceType) {
  if (!deviceType) return false
  const t = deviceType.toLowerCase()
  return t.includes('firewall') || t.includes('fw') || t.includes('palo')
}
