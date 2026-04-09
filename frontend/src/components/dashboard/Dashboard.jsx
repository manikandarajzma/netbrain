import { useEffect, useState, useCallback } from 'react'
import { fetchTopology } from '../../utils/api.js'
import styles from './Dashboard.module.css'

// Fixed positions for a triangle layout (3 devices)
const POSITIONS = {
  arista1: { x: 400, y: 120 },
  arista2: { x: 180, y: 380 },
  arista3: { x: 620, y: 380 },
}

const DEFAULT_POS = [
  { x: 400, y: 120 },
  { x: 180, y: 380 },
  { x: 620, y: 380 },
]

function getNodePos(device, index, total) {
  if (POSITIONS[device]) return POSITIONS[device]
  // Fallback: evenly spaced on a circle
  const angle = (2 * Math.PI * index) / total - Math.PI / 2
  return {
    x: 400 + 220 * Math.cos(angle),
    y: 260 + 220 * Math.sin(angle),
  }
}

function ospfStatusClass(device) {
  const full = device.ospf_full_count
  const total = device.ospf_neighbor_count
  if (total === 0) return styles.ospfNone
  if (full === total) return styles.ospfFull
  return styles.ospfPartial
}

function ospfLabel(device) {
  if (device.ospf_neighbor_count === 0) return 'No OSPF neighbors'
  if (device.ospf_full_count === device.ospf_neighbor_count)
    return `OSPF Full (${device.ospf_full_count})`
  return `OSPF ${device.ospf_full_count}/${device.ospf_neighbor_count} full`
}

export default function Dashboard() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const d = await fetchTopology()
      setData(d)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
    const interval = setInterval(load, 30000)
    return () => clearInterval(interval)
  }, [load])

  if (loading) return <div className={styles.state}>Loading topology…</div>
  if (error) return <div className={styles.stateError}>Error: {error}</div>
  if (!data) return null

  const { devices, links } = data
  const posMap = {}
  devices.forEach((d, i) => {
    posMap[d.hostname] = getNodePos(d.hostname, i, devices.length)
  })

  const selectedDevice = selected ? devices.find(d => d.hostname === selected) : null

  return (
    <div className={styles.dashboard}>
      <div className={styles.topRow}>
        <h2 className={styles.title}>Network Topology</h2>
        <button className={styles.refreshBtn} onClick={load} title="Refresh">↺ Refresh</button>
      </div>

      <div className={styles.content}>
        {/* SVG Topology */}
        <div className={styles.svgWrap}>
          <svg viewBox="0 0 800 520" className={styles.svg}>
            {/* Link edges */}
            {links.map((link, i) => {
              const a = posMap[link.device_a]
              const b = posMap[link.device_b]
              if (!a || !b) return null
              const mx = (a.x + b.x) / 2
              const my = (a.y + b.y) / 2
              const isHighlighted =
                selected && (link.device_a === selected || link.device_b === selected)
              return (
                <g key={i}>
                  <line
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    className={`${styles.link} ${isHighlighted ? styles.linkHighlighted : ''}`}
                  />
                  {/* Interface labels */}
                  <text
                    x={a.x + (mx - a.x) * 0.3}
                    y={a.y + (my - a.y) * 0.3 - 5}
                    className={styles.ifaceLabel}
                  >
                    {link.iface_a}
                  </text>
                  <text
                    x={b.x + (mx - b.x) * 0.3}
                    y={b.y + (my - b.y) * 0.3 - 5}
                    className={styles.ifaceLabel}
                  >
                    {link.iface_b}
                  </text>
                  {/* Midpoint subnet label */}
                  <text x={mx} y={my - 8} className={styles.subnetLabel}>
                    {link.ip_a}/31
                  </text>
                </g>
              )
            })}

            {/* Device nodes */}
            {devices.map((d) => {
              const pos = posMap[d.hostname]
              if (!pos) return null
              const isSelected = selected === d.hostname
              const fullOspf = d.ospf_full_count === d.ospf_neighbor_count && d.ospf_neighbor_count > 0
              const noOspf = d.ospf_neighbor_count === 0
              return (
                <g
                  key={d.hostname}
                  className={styles.node}
                  onClick={() => setSelected(isSelected ? null : d.hostname)}
                  style={{ cursor: 'pointer' }}
                >
                  <circle
                    cx={pos.x} cy={pos.y} r={44}
                    className={`${styles.nodeCircle} ${isSelected ? styles.nodeSelected : ''} ${noOspf ? styles.nodeWarn : fullOspf ? styles.nodeHealthy : styles.nodeDegraded}`}
                  />
                  <text x={pos.x} y={pos.y - 6} className={styles.nodeLabel}>
                    {d.hostname}
                  </text>
                  <text x={pos.x} y={pos.y + 10} className={styles.nodeSub}>
                    {d.platform === 'arista_eos' ? 'Arista EOS' : d.platform}
                  </text>
                  <text x={pos.x} y={pos.y + 24} className={styles.nodeSub}>
                    {d.route_count} routes
                  </text>
                </g>
              )
            })}
          </svg>
        </div>

        {/* Device cards */}
        <div className={styles.cards}>
          {devices.map(d => (
            <div
              key={d.hostname}
              className={`${styles.card} ${selected === d.hostname ? styles.cardSelected : ''}`}
              onClick={() => setSelected(selected === d.hostname ? null : d.hostname)}
            >
              <div className={styles.cardHeader}>
                <span className={styles.cardHostname}>{d.hostname}</span>
                <span className={`${styles.ospfBadge} ${ospfStatusClass(d)}`}>
                  {ospfLabel(d)}
                </span>
              </div>
              <div className={styles.cardStats}>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Platform</span>
                  <span className={styles.statValue}>{d.platform}</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Mgmt IP</span>
                  <span className={styles.statValue}>{d.mgmt_ip}</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>Routes</span>
                  <span className={styles.statValue}>{d.route_count}</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statLabel}>OSPF Neighbors</span>
                  <span className={styles.statValue}>
                    {d.ospf_neighbor_count === 0
                      ? '—'
                      : d.ospf_neighbors.join(', ')}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
