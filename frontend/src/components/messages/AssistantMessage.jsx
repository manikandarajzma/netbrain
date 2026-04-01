import { useMemo } from 'react'
import { classifyResponse } from '../../utils/responseClassifier.js'
import {
  isArrayOfObjects, isDeviceRackRow, DEVICE_RACK_KEYS,
  PANORAMA_COLUMN_ORDER, PANORAMA_TABLE_LABELS,
} from '../../utils/formatters.js'
import YesNoBadge from './YesNoBadge.jsx'
import MetricBadge from './MetricBadge.jsx'
import DirectAnswerBadge from './DirectAnswerBadge.jsx'
import ErrorMessage from './ErrorMessage.jsx'
import JsonFallback from './JsonFallback.jsx'
import MarkdownContent from './MarkdownContent.jsx'
import PathVisualization from '../path/PathVisualization.jsx'
import DataTable from '../tables/DataTable.jsx'
import VerticalTable from '../tables/VerticalTable.jsx'
import BatchResults from '../tables/BatchResults.jsx'
import MemoryFeedback from './MemoryFeedback.jsx'
import InterfaceCounters from './InterfaceCounters.jsx'
import styles from './AssistantMessage.module.css'

export default function AssistantMessage({ content, memories }) {
  const classified = useMemo(() => classifyResponse(content), [content])

  const hasYesNo = typeof content === 'object' && content?.yes_no_answer
  const hasMetric = typeof content === 'object' && content?.metric_answer
  const hasDirectAnswer = typeof content === 'object' && content?.direct_answer
  const followUp = typeof content === 'object' && content?.follow_up
  const insights = typeof content === 'object' && Array.isArray(content?.insights) && content.insights.length > 0 ? content.insights : null

  const structuredText = useMemo(() => {
    if (classified.type !== 'structured') return null
    const c = content
    const msg = c.message || (c.ai_analysis && (c.ai_analysis.summary || c.ai_analysis.Summary))
    return msg && (typeof msg === 'string' ? msg : JSON.stringify(msg))
  }, [classified.type, content])

  const pathSummaryDesc = useMemo(() => {
    if (classified.type !== 'path-summary') return ''
    const c = content
    const desc = c.path_status_description || c.path_status || c.statusDescription || ''
    const noise = ['l2 connections has not been discovered', 'l2 connection has not been discovered']
    if (noise.some(p => desc.toLowerCase().includes(p))) return ''
    return desc
  }, [classified.type, content])

  const tableGroups = useMemo(() => {
    const c = content
    if (!c || typeof c !== 'object' || Array.isArray(c)) {
      if (Array.isArray(c) && isArrayOfObjects(c)) {
        if (c.length === 1 && isDeviceRackRow(c[0])) {
          return [{ type: 'vertical', row: c[0], keys: DEVICE_RACK_KEYS }]
        }
        return [{ type: 'horizontal', rows: c }]
      }
      return []
    }

    if (isDeviceRackRow(c)) {
      return [{ type: 'vertical', row: c, keys: DEVICE_RACK_KEYS }]
    }

    const groups = []
    const arrayKeys = Object.keys(c).filter(k => {
      const v = c[k]
      return Array.isArray(v) && v.length > 0 && v.every(x => x != null && typeof x === 'object' && !Array.isArray(x))
    })
    const BADGE_KEYS = new Set(['yes_no_answer', 'metric_answer', 'direct_answer'])
    const HIDDEN_KEYS = new Set(['desc_units', 'outer_width', 'outer_unit', 'outer_depth', 'intent', 'format', 'vsys', 'queried_ip', 'follow_up', 'follow_up_action', 'insights', 'interface_counters', 'source', 'destination'])
    const flatKeys = Object.keys(c).filter(k => {
      if (arrayKeys.includes(k) || k === 'ai_analysis' || BADGE_KEYS.has(k) || HIDDEN_KEYS.has(k)) return false
      const v = c[k]
      return v != null && typeof v !== 'object' && (typeof v !== 'string' || v.length <= 500)
    })

    if (flatKeys.length > 0 && flatKeys.length <= 25 && arrayKeys.length > 0) {
      const flatLower = flatKeys.map(k => k.toLowerCase())
      const hasRack = flatLower.includes('site') || flatLower.includes('facility') || (arrayKeys.includes('devices') && flatLower.some(k => k === 'name' || k === 'rack_name'))
      const looksPanorama = (flatLower.includes('ip_address') || flatLower.includes('vsys')) && (arrayKeys.includes('address_objects') || arrayKeys.includes('address_groups'))
      if (hasRack && !looksPanorama) {
        groups.push({
          type: 'horizontal',
          rows: [Object.fromEntries(flatKeys.map(k => [k, c[k]]))],
          heading: arrayKeys.length > 0 ? 'Rack details' : null,
        })
      }
    }

    const isPanorama = arrayKeys.includes('address_objects') || arrayKeys.includes('address_groups') || arrayKeys.includes('policies') || arrayKeys.includes('members')
      || arrayKeys.includes('orphaned_address_objects') || arrayKeys.includes('unused_address_groups')
    const tableOrder = isPanorama
      ? ['members', 'address_objects', 'address_groups', 'policies', 'orphaned_address_objects', 'unused_address_groups']
      : arrayKeys

    for (const key of tableOrder) {
      if (!arrayKeys.includes(key) || key === 'path_hops' || key === 'reverse_path_hops') continue
      const arr = c[key]
      const heading = PANORAMA_TABLE_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, ch => ch.toUpperCase())
      const colOrder = PANORAMA_COLUMN_ORDER[key] || null
      if (arr.length === 1 && isDeviceRackRow(arr[0])) {
        groups.push({ type: 'vertical', row: arr[0], keys: DEVICE_RACK_KEYS, heading })
      } else {
        groups.push({ type: 'horizontal', rows: arr, columns: colOrder, heading })
      }
    }

    if (groups.length === 0) {
      const scalarKeys = Object.keys(c).filter(k => c[k] != null && typeof c[k] !== 'object' && !BADGE_KEYS.has(k))
      if (scalarKeys.length > 0 && scalarKeys.length <= 20) {
        groups.push({ type: 'horizontal', rows: [Object.fromEntries(scalarKeys.map(k => [k, c[k]]))] })
      }
    }

    return groups
  }, [content])

  // String content — render as markdown so tool responses with headings,
  // bold, code blocks and horizontal rules display correctly.
  if (classified.type === 'text') {
    return (
      <>
        <MarkdownContent text={classified.content} />
        <InterfaceCounters counters={content?.interface_counters} />
        <MemoryFeedback memories={memories} />
      </>
    )
  }

  // Object content
  return (
    <>
      {hasYesNo && <YesNoBadge text={content.yes_no_answer} />}
      {hasMetric && <MetricBadge text={content.metric_answer} />}
      {classified.type === 'path' && (
        <>
          <p className={styles.summaryHeading}>Path Summary</p>
          <PathVisualization content={content} />
          {content.reverse_path_hops && content.reverse_path_hops.length > 0 && (
            <>
              <p className={styles.summaryHeading} style={{ marginTop: '1.25rem' }}>Return Path</p>
              <PathVisualization content={{ ...content, path_hops: content.reverse_path_hops, source: content.destination, destination: content.source }} />
            </>
          )}
        </>
      )}

      {hasDirectAnswer && <DirectAnswerBadge text={content.direct_answer} />}

      {classified.type === 'batch' && (
        <BatchResults results={content.batch_results} tool={content.tool || ''} />
      )}

      {classified.type === 'error' && <ErrorMessage content={content} />}

      {classified.type === 'clarification' && (
        <>
          <p>{content.error}</p>
          <p className={styles.clarificationPrompt}>Which site? {content.sites.join(', ')}. Reply with the site name.</p>
        </>
      )}

      {classified.type === 'path-summary' && (
        <div className={styles.pathVisual}>
          <p className={styles.pathStatus}>Path: {content.source} {'\u2192'} {content.destination}</p>
          {pathSummaryDesc && <p className={styles.pathStatus} style={{ marginTop: '0.5rem' }}>{pathSummaryDesc}</p>}
          {content.message ? <p className={styles.pathNote}>{content.message}</p>
            : content.note ? <p className={styles.pathNote}>{content.note}</p> : null}
          {content.status === 'denied' && (content.firewall_denied_by || content.policy_details) && (
            <>
              {content.firewall_denied_by && <p className={styles.denyText}>Denied by firewall: {content.firewall_denied_by}</p>}
              {content.policy_details && <p className={styles.policyText}>Policy: {content.policy_details}</p>}
            </>
          )}
        </div>
      )}

      {classified.type === 'structured' && (
        <>
          {structuredText && tableGroups.length === 0 && <p style={{ marginBottom: '0.75rem' }}>{structuredText}</p>}
          {tableGroups.map((group, gi) => (
            <div key={gi}>
              {group.heading && <p className={styles.summaryHeading}>{group.heading}</p>}
              {group.type === 'vertical'
                ? <VerticalTable row={group.row} preferredKeys={group.keys} />
                : <DataTable rows={group.rows} columns={group.columns || null} />
              }
            </div>
          ))}
        </>
      )}

      {classified.type === 'table' && (
        <>
          {tableGroups.map((group, gi) => (
            <div key={gi}>
              {group.heading && <p className={styles.summaryHeading}>{group.heading}</p>}
              {group.type === 'vertical'
                ? <VerticalTable row={group.row} preferredKeys={group.keys} />
                : <DataTable rows={group.rows} columns={group.columns || null} />
              }
            </div>
          ))}
        </>
      )}

      {classified.type === 'multi' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {content.multi_results.map((result, i) => (
            <AssistantMessage key={i} content={result} />
          ))}
        </div>
      )}

      {classified.type === 'json' && <JsonFallback content={content} />}

      {insights && (
        <div style={{ marginTop: '0.75rem', padding: '0.6rem 0.85rem', background: 'rgba(249,226,175,0.08)', borderLeft: '3px solid #f9e2af', borderRadius: '6px' }}>
          <p style={{ fontSize: '0.8rem', fontWeight: 600, color: '#f9e2af', marginBottom: insights.length > 1 ? '0.35rem' : 0 }}>Insights</p>
          {insights.map((insight, i) => (
            <p key={i} style={{ fontSize: '0.85rem', color: '#cdd6f4', margin: 0, marginTop: i > 0 ? '0.25rem' : 0 }}>{insight}</p>
          ))}
        </div>
      )}
      {followUp && <p style={{ marginTop: '0.75rem', fontStyle: 'italic', opacity: 0.6, fontSize: '0.875rem' }}>{followUp}</p>}
      <InterfaceCounters counters={content?.interface_counters} />
      <MemoryFeedback memories={memories} />
    </>
  )
}
