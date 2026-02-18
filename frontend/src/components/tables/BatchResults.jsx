import { useMemo } from 'react'
import PathVisualization from '../path/PathVisualization.jsx'
import { downloadCsv } from '../../utils/csvExport.js'
import styles from './BatchResults.module.css'

export default function BatchResults({ results, tool }) {
  const isPathQuery = tool === 'query_network_path'

  const counts = useMemo(() => {
    const c = { allowed: 0, denied: 0, unknown: 0, error: 0, success: 0, failed: 0 }
    results.forEach(r => {
      const s = (r.status || '').toLowerCase()
      if (s in c) c[s]++
      else c.unknown++
    })
    return c
  }, [results])

  const headers = isPathQuery
    ? ['Source', 'Destination', 'Protocol', 'Port', 'Status', 'Path', 'Details']
    : ['Source', 'Destination', 'Protocol', 'Port', 'Status', 'Reason', 'Firewall', 'Policy']

  function statusClass(status) {
    const s = (status || 'unknown').toLowerCase()
    if (s === 'success') return 'allowed'
    if (s === 'failed') return 'denied'
    return s
  }

  function exportCsv() {
    const rows = results.map(r =>
      isPathQuery
        ? [r.source, r.destination, r.protocol, r.port, r.status, r.path_summary, r.reason]
        : [r.source, r.destination, r.protocol, r.port, r.status, r.reason, r.firewall_denied_by, r.policy_details]
    )
    downloadCsv(headers, rows, 'batch_results.csv')
  }

  const pathResults = isPathQuery ? results.filter(r => r.path_hops && r.path_hops.length > 0) : []

  return (
    <div>
      {/* Summary */}
      <div className={styles.batchSummary}>
        {isPathQuery ? (
          <>
            <span className={`${styles.batchStat} ${styles.allowed}`}>{counts.success} Success</span>
            <span className={`${styles.batchStat} ${styles.denied}`}>{counts.failed} Failed</span>
            {counts.error > 0 && <span className={`${styles.batchStat} ${styles.error}`}>{counts.error} Error</span>}
            {counts.unknown > 0 && <span className={`${styles.batchStat} ${styles.unknown}`}>{counts.unknown} Unknown</span>}
          </>
        ) : (
          <>
            <span className={`${styles.batchStat} ${styles.allowed}`}>{counts.allowed} Allowed</span>
            <span className={`${styles.batchStat} ${styles.denied}`}>{counts.denied} Denied</span>
            <span className={`${styles.batchStat} ${styles.unknown}`}>{counts.unknown} Unknown</span>
            {counts.error > 0 && <span className={`${styles.batchStat} ${styles.error}`}>{counts.error} Error</span>}
          </>
        )}
        <span className={styles.batchTotal}> out of {results.length} total</span>
      </div>

      {/* Results table */}
      <div style={{ overflowX: 'auto' }}>
        <table className={styles.batchTable}>
          <thead>
            <tr>{headers.map(h => <th key={h}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {results.map((r, i) => (
              <tr key={i}>
                <td>{r.source}</td>
                <td>{r.destination}</td>
                <td>{(r.protocol || 'tcp').toUpperCase()}</td>
                <td>{r.port || '0'}</td>
                <td className={`${styles.statusCell} ${styles[statusClass(r.status)] || ''}`}>{(r.status || 'unknown').toLowerCase()}</td>
                {isPathQuery ? (
                  <>
                    <td className={styles.wrapCell}>{r.path_summary || ''}</td>
                    <td className={styles.wrapCell}>{r.reason || ''}</td>
                  </>
                ) : (
                  <>
                    <td className={styles.wrapCell}>{r.reason || ''}</td>
                    <td>{r.firewall_denied_by || ''}</td>
                    <td className={styles.wrapCell}>{r.policy_details || ''}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <button type="button" className={styles.exportCsv} onClick={exportCsv}>Export Results CSV</button>

      {/* Per-row path visualizations */}
      {pathResults.map((r, i) => (
        <div key={'path-' + i} className={styles.batchPathSection}>
          <div className={styles.batchPathHeader}>
            <h4 className={styles.batchPathTitle}>{r.source} {'\u2192'} {r.destination}</h4>
            <span className={`${styles.batchStatusBadge} ${styles[statusClass(r.status)] || ''}`}>{(r.status || '').toLowerCase()}</span>
          </div>
          <PathVisualization content={{
            path_hops: r.path_hops,
            source: r.source,
            destination: r.destination,
            path_status: r.path_status || r.status,
            path_failure_reason: r.path_failure_reason || '',
            path_status_description: r.reason || '',
          }} />
        </div>
      ))}
    </div>
  )
}
