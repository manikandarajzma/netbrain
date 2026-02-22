export function normalizeInterface(s) {
  if (s == null || s === '') return ''
  s = String(s).trim()
  if (/^ethernet/i.test(s)) return 'Ethernet' + s.slice(8)
  if (/^eth/i.test(s)) return 'Ethernet' + s.slice(3)
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function cellText(val) {
  if (val == null) return ''
  if (typeof val === 'string' && val.trim().charAt(0) === '{') {
    try {
      const obj = JSON.parse(val)
      if (obj && typeof obj === 'object') {
        const name = (obj.intfDisplaySchemaObj && obj.intfDisplaySchemaObj.value) || obj.PhysicalInftName || obj.name || obj.value
        if (name != null) return String(name)
      }
    } catch {}
  }
  if (Array.isArray(val)) {
    if (val.length === 0) return '\u2014'
    if (val.every(x => x == null || typeof x !== 'object'))
      return val.map(x => x == null ? '' : String(x)).join(', ')
    return val.map(item => {
      if (item == null) return ''
      if (typeof item !== 'object') return String(item)
      if (item.name != null && item.value != null) return item.name + ' (' + item.value + ')'
      if (item.name != null) return item.name
      const parts = []
      for (const k in item) {
        if (Object.prototype.hasOwnProperty.call(item, k) && item[k] != null && typeof item[k] !== 'object') parts.push(item[k])
      }
      return parts.length ? parts.join(', ') : JSON.stringify(item)
    }).join('; ')
  }
  if (typeof val === 'object') {
    const keys = Object.keys(val).filter(k => val[k] != null && typeof val[k] !== 'object')
    if (keys.length <= 3) return keys.map(k => val[k]).join(', ')
    return keys.slice(0, 3).map(k => k + ': ' + val[k]).join('; ')
  }
  return String(val)
}

export function isArrayOfObjects(val) {
  return Array.isArray(val) && val.length > 0 && val.every(x => x != null && typeof x === 'object' && !Array.isArray(x))
}

export const PANORAMA_COLUMN_ORDER = {
  address_objects: ['name', 'type', 'value', 'location', 'device_group'],
  address_groups: ['name', 'contains_address_object', 'members', 'location', 'device_group'],
  members: ['name', 'type', 'value', 'location', 'device_group'],
  policies: ['name', 'type', 'rulebase', 'action', 'source', 'destination', 'services', 'address_groups', 'address_objects', 'location', 'device_group'],
}

export const PANORAMA_TABLE_LABELS = {
  address_objects: 'Address objects',
  address_groups: 'Address groups',
  members: 'Address group members (IPs)',
  policies: 'Policy details',
}

export const DEVICE_RACK_KEYS = ['device', 'rack', 'position', 'face', 'site', 'status', 'device_type']

export function isDeviceRackRow(row) {
  if (!row || typeof row !== 'object') return false
  let has = 0
  for (const k of DEVICE_RACK_KEYS) {
    if (Object.prototype.hasOwnProperty.call(row, k)) has++
  }
  return has >= 4
}

const _BADGE_KEYS = new Set(['yes_no_answer', 'metric_answer', 'direct_answer'])
const _HIDDEN_KEYS = new Set(['desc_units', 'outer_width', 'outer_unit', 'outer_depth', 'intent', 'format', 'vsys', 'queried_ip'])

export function orderKeys(row, preferredKeys) {
  const skip = k => k.startsWith('_debug') || k === 'ai_analysis' || _BADGE_KEYS.has(k) || _HIDDEN_KEYS.has(k)
  if (preferredKeys && preferredKeys.length) {
    return [
      ...preferredKeys.filter(k => Object.prototype.hasOwnProperty.call(row, k)),
      ...Object.keys(row).filter(k => !preferredKeys.includes(k) && !skip(k)),
    ]
  }
  return Object.keys(row).filter(k => !skip(k))
}
