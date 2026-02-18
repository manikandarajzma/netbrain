import { useMemo } from 'react'
import DeviceIcon from './DeviceIcon.jsx'
import { normalizeInterface } from '../../utils/formatters.js'
import { isFirewallType } from '../../utils/deviceIcons.js'
import styles from './PathItem.module.css'

export default function PathItem({ node, sourceIp, destIp }) {
  const inInt = node.in != null ? normalizeInterface(node.in) : ''
  const outInt = node.out != null ? normalizeInterface(node.out) : ''

  const intfText = useMemo(() => {
    const parts = []
    if (inInt) parts.push('In: ' + inInt)
    if (outInt) parts.push('Out: ' + outInt)
    return parts.join(' | ')
  }, [inInt, outInt])

  const displayName = useMemo(() => {
    let name = (node.name && node.name !== 'Unknown') ? node.name : null
    if (!name && node.isDest && destIp) name = destIp
    if (!name && node.isSource && sourceIp) name = sourceIp
    return name || 'Device'
  }, [node, sourceIp, destIp])

  const showIp = node.isSource && sourceIp ? sourceIp : node.isDest && destIp ? destIp : null

  const isFirewall = isFirewallType(node.type)

  const zonesText = useMemo(() => {
    if (!isFirewall) return ''
    const parts = []
    if (node.in_zone) parts.push('In: ' + node.in_zone)
    if (node.out_zone) parts.push('Out: ' + node.out_zone)
    if (node.dg) parts.push('DG: ' + node.dg)
    return parts.join(' | ')
  }, [isFirewall, node])

  return (
    <div className={`${styles.pathItem} path-item`}>
      {node.isSource && <span className={`${styles.pathBadge} ${styles.source}`}>A</span>}
      {node.isDest && <span className={`${styles.pathBadge} ${styles.dest}`}>B</span>}
      <DeviceIcon deviceType={node.type} />
      <div className={styles.pathNodeBody}>
        <div className={styles.pathDeviceName}>{displayName}</div>
        {showIp && <div className={styles.pathIp}>{showIp}</div>}
        {intfText && <div className={styles.pathInterfaces}>{intfText}</div>}
        {zonesText && <div className={styles.pathZones}>{zonesText}</div>}
      </div>
    </div>
  )
}
