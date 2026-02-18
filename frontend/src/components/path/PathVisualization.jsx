import { useState, useMemo } from 'react'
import PathItem from './PathItem.jsx'
import PathConnectors from './PathConnectors.jsx'
import FirewallDetails from './FirewallDetails.jsx'
import PathFullscreen from './PathFullscreen.jsx'
import { isFirewallType } from '../../utils/deviceIcons.js'
import styles from './PathVisualization.module.css'

export default function PathVisualization({ content }) {
  const [showFullscreen, setShowFullscreen] = useState(false)

  const pathStatus = (content.path_status || '').toLowerCase()
  const contentStatus = (content.status || '').toLowerCase()
  const showStatusBar = pathStatus === 'failed' || contentStatus === 'denied' || contentStatus === 'unknown'
  const statusText = content.reason || content.path_failure_reason || content.path_status_description || ''
  const statusFallback = 'Path: ' + (content.source || '') + ' \u2192 ' + (content.destination || '')
  const showDenyBlock = content.firewall_denied_by || content.policy_details

  const nodes = useMemo(() => {
    const hops = content.path_hops
    if (!hops || !hops.length) return []

    const h0 = hops[0]
    const result = [{
      name: h0.from_device,
      type: h0.from_device_type,
      in: null,
      out: h0.out_interface,
      isSource: true,
      in_zone: null,
      out_zone: h0.out_zone,
      dg: h0.device_group,
    }]

    const seen = new Set([h0.from_device])
    for (const hop of hops) {
      if (hop.to_device && !seen.has(hop.to_device)) {
        seen.add(hop.to_device)
        result.push({
          name: hop.to_device,
          type: hop.to_device_type,
          in: hop.in_interface,
          out: hop.out_interface,
          isDest: false,
          in_zone: hop.in_zone,
          out_zone: hop.out_zone,
          dg: hop.device_group,
        })
      }
    }
    if (result.length > 1) {
      result[result.length - 1].isDest = true
    }
    return result
  }, [content])

  const firewalls = useMemo(() =>
    nodes.filter(n => isFirewallType(n.type))
  , [nodes])

  const renderPath = (keyPrefix = '', isFullscreen = false) => (
    <div className={styles.pathVisualRowWrap}>
      <div className={`${styles.pathHorizontal} ${isFullscreen ? styles.fullscreenPath : ''}`}>
        <PathConnectors itemCount={nodes.length} />
        {nodes.map((node, i) => (
          <PathItem
            key={(keyPrefix || '') + i}
            node={node}
            sourceIp={content.source || ''}
            destIp={content.destination || ''}
          />
        ))}
      </div>
    </div>
  )

  return (
    <div className={styles.pathVisual}>
      {showStatusBar && (
        <p className={styles.pathStatusBar}>{statusText || statusFallback}</p>
      )}
      {showDenyBlock && (
        <div className={styles.pathDenyDetails}>
          {content.firewall_denied_by && <p className={styles.denyFw}>Denied by firewall: {content.firewall_denied_by}</p>}
          {content.policy_details && <p className={styles.denyPolicy}>Policy: {content.policy_details}</p>}
        </div>
      )}
      <button className={styles.pathExpandBtn} onClick={() => setShowFullscreen(true)} title="Open fullscreen view">
        &#x26F6; Expand
      </button>
      {renderPath()}
      {firewalls.length > 0 && <FirewallDetails firewalls={firewalls} />}
      {showFullscreen && (
        <PathFullscreen onClose={() => setShowFullscreen(false)}>
          <div className={styles.pathVisualClone}>
            {renderPath('fs-', true)}
          </div>
        </PathFullscreen>
      )}
    </div>
  )
}
