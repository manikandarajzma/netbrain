import styles from './MemoryFeedback.module.css'

function parseIncidentNumber(resultSummary) {
  const m = (resultSummary || '').match(/^\[([A-Z]+\d+)\]\s*/)
  return m ? m[1] : null
}

function stripIncidentNumber(resultSummary) {
  return (resultSummary || '').replace(/^\[[A-Z]+\d+\]\s*/, '')
}

export default function MemoryFeedback({ memories }) {
  if (!memories || memories.length === 0) return null
  return (
    <div className={styles.container}>
      <p className={styles.heading}>📚 Past cases recalled as context</p>
      <div className={styles.entries}>
        {memories.map((m, i) => {
          const days = Math.max(0, Math.floor((Date.now() / 1000 - (m.timestamp || 0)) / 86400))
          const ageStr = days === 0 ? 'today' : `${days}d ago`
          const isDevice = m.match_type === 'device'
          const simPct = Math.round((m.similarity || 0) * 100)
          const incNumber = parseIncidentNumber(m.result_summary)
          const resolution = (m.resolution || '').trim()

          return (
            <div key={i} className={`${styles.entry} ${isDevice ? styles.entryDevice : styles.entrySemantic}`}>
              <div className={styles.entryHeader}>
                <span className={styles.title}>{m.query}</span>
                <div className={styles.tags}>
                  {incNumber && <span className={styles.incBadge}>{incNumber}</span>}
                  <span className={`${styles.matchBadge} ${isDevice ? styles.matchDevice : styles.matchSemantic}`}>
                    {isDevice
                      ? (simPct > 0 ? `🔗 on-path · ${simPct}%` : '🔗 on-path device')
                      : `⬡ ${simPct}% similar`}
                  </span>
                  <span className={styles.age}>{ageStr}</span>
                </div>
              </div>
              {resolution
                ? <p className={styles.rootCause}><span className={styles.resolutionLabel}>Resolution: </span>{resolution}</p>
                : <p className={`${styles.rootCause} ${styles.noResolution}`}>No resolution notes — incident is open</p>
              }
            </div>
          )
        })}
      </div>
    </div>
  )
}
