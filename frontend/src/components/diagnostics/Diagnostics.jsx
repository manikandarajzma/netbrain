import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchDiagnostics } from '../../utils/api.js'
import styles from './Diagnostics.module.css'

function toneForStatus(status) {
  if (status === 'healthy' || status === 'ok' || status === true) return 'healthy'
  if (status === 'degraded' || status === 'model_not_found' || status === 'pending') return 'degraded'
  return 'unhealthy'
}

function prettyServiceName(name) {
  return name === 'mcp' ? 'MCP' : name === 'ollama' ? 'Ollama' : name === 'nornir' ? 'Nornir' : name
}

function MetricPill({ label, value, tone = 'default' }) {
  return (
    <div className={`${styles.metricPill} ${styles[tone] || ''}`}>
      <span className={styles.metricPillLabel}>{label}</span>
      <span className={styles.metricPillValue}>{value}</span>
    </div>
  )
}

function ServiceCard({ name, service = {} }) {
  const tone = toneForStatus(service.status)
  const details = []

  if (name === 'mcp' && service.tools_registered != null) {
    details.push(`tools ${service.tools_registered}`)
  }
  if (name === 'ollama') {
    const models = service.models || {}
    const uniqueModels = [...new Set(Object.values(models).filter(Boolean))]
    if (uniqueModels.length === 1) {
      details.push(uniqueModels[0])
    } else if (uniqueModels.length > 1) {
      details.push(`${uniqueModels.length} models`)
    }
    if (Array.isArray(service.missing_models) && service.missing_models.length > 0) {
      details.push(`missing ${service.missing_models.length}`)
    }
  }
  if (name === 'nornir') {
    details.push(service.device_count != null ? `${service.device_count} devices` : 'device count unavailable')
  }
  if (service.url) {
    details.push(service.url.replace(/^https?:\/\//, ''))
  }

  return (
    <article className={`${styles.serviceCard} ${styles[tone] || ''}`}>
      <div className={styles.serviceCardHeader}>
        <div>
          <div className={styles.serviceName}>{prettyServiceName(name)}</div>
          <div className={styles.serviceDetails}>{details.join(' • ')}</div>
        </div>
        <span className={`${styles.statusBadge} ${styles[tone] || ''}`}>{service.status || 'unknown'}</span>
      </div>
    </article>
  )
}

function SimpleTable({ columns, rows, emptyMessage }) {
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            {columns.map(column => <th key={column.key}>{column.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td className={styles.emptyCell} colSpan={columns.length}>{emptyMessage}</td>
            </tr>
          ) : rows.map((row, index) => (
            <tr key={row.id || index}>
              {columns.map(column => (
                <td key={column.key} className={column.mono ? styles.mono : ''}>
                  {row[column.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Diagnostics() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await fetchDiagnostics())
    } catch (e) {
      setError(e.message || 'Failed to load diagnostics')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const viewModel = useMemo(() => {
    if (!data) return null

    const owners = Object.entries(data.owners || {})
  const profiles = Object.entries(data.tools?.profiles || {})
  const registeredTools = data.tools?.registered || []
  const counters = data.metrics?.counters || []
  const timings = data.metrics?.timings || []
  const health = data.health || {}
  const overallHealth = health.overall || {}
  const services = Object.entries(health.services || {})
  const checkpointer = data.runtime?.checkpointer || {}
  const checkpointerState = checkpointer.state || 'pending'
  const checkpointerLabel = checkpointer.label || 'Pending first graph run'
  const counterTotal = counters.reduce((sum, item) => sum + (item.value || 0), 0)
    const timingSamples = timings.reduce((sum, item) => sum + (item.count || 0), 0)

    const topCounters = [...counters]
      .sort((a, b) => (b.value || 0) - (a.value || 0))
      .slice(0, 5)
      .map(counter => ({
        id: `${counter.name}:${JSON.stringify(counter.tags || {})}`,
        name: counter.name,
        value: counter.value,
        tags: JSON.stringify(counter.tags || {}),
      }))

    const topTimings = [...timings]
      .sort((a, b) => (b.avg_ms || 0) - (a.avg_ms || 0))
      .slice(0, 5)
      .map(timing => ({
        id: `${timing.name}:${JSON.stringify(timing.tags || {})}`,
        name: timing.name,
        avg_ms: timing.avg_ms,
        max_ms: timing.max_ms,
        count: timing.count,
      }))

    return {
      owners,
      profiles,
      registeredTools,
      counters,
      timings,
      health,
      overallHealth,
      services,
      checkpointer,
      checkpointerState,
      checkpointerLabel,
      counterTotal,
      timingSamples,
      topCounters,
      topTimings,
    }
  }, [data])

  if (loading) return <div className={styles.state}>Loading diagnostics…</div>
  if (error) return <div className={styles.stateError}>Error: {error}</div>
  if (!viewModel) return null

  const {
    owners,
    profiles,
    registeredTools,
    overallHealth,
    services,
    checkpointer,
    checkpointerState,
    checkpointerLabel,
    counterTotal,
    timingSamples,
    topCounters,
    topTimings,
  } = viewModel

  const overallTone = toneForStatus(overallHealth.status)

  return (
    <div className={styles.diagnostics}>
      <div className={styles.topRow}>
        <div>
          <h2 className={styles.title}>Diagnostics</h2>
          <p className={styles.subtitle}>Live backend health first, architecture details second.</p>
        </div>
        <button className={styles.refreshBtn} onClick={load} title="Refresh diagnostics">↺ Refresh</button>
      </div>

      <section className={`${styles.hero} ${styles[overallTone] || ''}`}>
        <div className={styles.heroMain}>
          <div className={styles.heroEyebrow}>Overall health</div>
          <div className={styles.heroTitle}>{overallHealth.label || 'Unknown'}</div>
          <div className={styles.heroSubtitle}>
            {services.length} services checked • checkpointer {checkpointerLabel.toLowerCase()}
          </div>
        </div>
        <div className={styles.heroMetrics}>
          <MetricPill label="Services" value={services.length} tone={overallTone} />
          <MetricPill label="Counter events" value={counterTotal} />
          <MetricPill label="Timing samples" value={timingSamples} />
          <MetricPill label="Profiles" value={profiles.length} />
        </div>
      </section>

      <section>
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>Service health</h3>
          <span className={styles.sectionHint}>These are the live dependencies the app currently relies on.</span>
        </div>
        <div className={styles.serviceGrid}>
          {services.map(([name, service]) => (
            <ServiceCard key={name} name={name} service={service} />
          ))}
        </div>
      </section>

      <div className={styles.summaryGrid}>
        <section className={styles.panel}>
          <div className={styles.panelHeader}>
            <h3 className={styles.sectionTitle}>Runtime signals</h3>
            <span className={styles.sectionHint}>Short operational view</span>
          </div>
          <div className={styles.signalList}>
            <div className={styles.signalRow}>
              <span className={styles.signalLabel}>Checkpointer</span>
              <span className={`${styles.statusBadge} ${styles[toneForStatus(checkpointerState)] || ''}`}>
                {checkpointerLabel}
              </span>
            </div>
            <div className={styles.signalRow}>
              <span className={styles.signalLabel}>Registered tools</span>
              <span className={styles.signalValue}>{registeredTools.length}</span>
            </div>
            <div className={styles.signalRow}>
              <span className={styles.signalLabel}>Owners</span>
              <span className={styles.signalValue}>{owners.length}</span>
            </div>
            <div className={styles.signalRow}>
              <span className={styles.signalLabel}>Metric counters</span>
              <span className={styles.signalValue}>{counterTotal}</span>
            </div>
            <div className={styles.signalRow}>
              <span className={styles.signalLabel}>Timing samples</span>
                <span className={styles.signalValue}>{timingSamples}</span>
              </div>
              {checkpointer.error ? (
                <div className={styles.inlineNote}>
                  Running without Redis persistence: {checkpointer.error}
                </div>
              ) : null}
            </div>
          </section>

        <section className={styles.panel}>
          <div className={styles.panelHeader}>
            <h3 className={styles.sectionTitle}>Hot metrics</h3>
            <span className={styles.sectionHint}>Top timings and counters</span>
          </div>
          <div className={styles.hotMetrics}>
            <div className={styles.hotMetricBlock}>
              <div className={styles.blockTitle}>Top timings</div>
              {topTimings.length === 0 ? (
                <div className={styles.emptyNote}>No timing samples yet.</div>
              ) : topTimings.map(item => (
                <div key={item.id} className={styles.hotRow}>
                  <span className={styles.hotLabel}>{item.name}</span>
                  <span className={styles.hotValue}>{item.avg_ms} ms avg</span>
                </div>
              ))}
            </div>
            <div className={styles.hotMetricBlock}>
              <div className={styles.blockTitle}>Top counters</div>
              {topCounters.length === 0 ? (
                <div className={styles.emptyNote}>No counter activity yet.</div>
              ) : topCounters.map(item => (
                <div key={item.id} className={styles.hotRow}>
                  <span className={styles.hotLabel}>{item.name}</span>
                  <span className={styles.hotValue}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>

      <details className={styles.disclosure}>
        <summary className={styles.disclosureSummary}>Advanced architecture details</summary>
        <div className={styles.disclosureBody}>
          <div className={styles.summaryGrid}>
            <section className={styles.panel}>
              <div className={styles.panelHeader}>
                <h3 className={styles.sectionTitle}>Owners</h3>
                <span className={styles.sectionHint}>Runtime ownership map</span>
              </div>
              <div className={styles.ownerList}>
                {owners.map(([key, value]) => (
                  <div key={key} className={styles.ownerRow}>
                    <span className={styles.ownerKey}>{key}</span>
                    <span className={styles.ownerValue}>{value}</span>
                  </div>
                ))}
              </div>
            </section>

            <section className={styles.panel}>
              <div className={styles.panelHeader}>
                <h3 className={styles.sectionTitle}>Profiles</h3>
                <span className={styles.sectionHint}>Capability-level routing surface</span>
              </div>
              <SimpleTable
                columns={[
                  { key: 'name', label: 'Profile', mono: true },
                  { key: 'capabilities', label: 'Capabilities' },
                ]}
                rows={profiles.map(([name, capabilities]) => ({
                  id: name,
                  name,
                  capabilities: capabilities.join(', '),
                }))}
                emptyMessage="No profiles available."
              />
            </section>
          </div>

          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h3 className={styles.sectionTitle}>Registered tools</h3>
              <span className={styles.sectionHint}>Uniform agent-facing interface</span>
            </div>
            <SimpleTable
              columns={[
                { key: 'name', label: 'Name', mono: true },
                { key: 'module', label: 'Module', mono: true },
                { key: 'capabilities', label: 'Capabilities' },
              ]}
              rows={registeredTools.map(tool => ({
                id: `${tool.module}:${tool.name}`,
                name: tool.name,
                module: tool.module,
                capabilities: (tool.capabilities || []).join(', '),
              }))}
              emptyMessage="No registered tools."
            />
          </section>
        </div>
      </details>
    </div>
  )
}
