import { useState } from 'react'
import { correctMemory } from '../../utils/api.js'
import styles from './MemoryFeedback.module.css'

function MemoryEntry({ memory }) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(memory.result_summary || '')
  const [status, setStatus] = useState(null) // null | 'saving' | 'saved' | 'error'

  const ageStr = (() => {
    const days = Math.max(0, Math.floor((Date.now() / 1000 - (memory.timestamp || 0)) / 86400))
    return days === 0 ? 'today' : `${days}d ago`
  })()
  const simPct = Math.round((memory.similarity || 0) * 100)

  async function handleSave() {
    if (!value.trim()) return
    setStatus('saving')
    try {
      await correctMemory(memory.query, value.trim())
      setStatus('saved')
      setEditing(false)
    } catch {
      setStatus('error')
    }
  }

  return (
    <div className={styles.entry}>
      <div className={styles.meta}>
        <span className={styles.badge}>{ageStr} · {simPct}% similar</span>
        <span className={styles.query}>{memory.query}</span>
      </div>

      {editing ? (
        <div className={styles.editArea}>
          <textarea
            className={styles.textarea}
            value={value}
            onChange={e => setValue(e.target.value)}
            rows={3}
            autoFocus
          />
          <div className={styles.editActions}>
            <button className={styles.cancelBtn} onClick={() => { setEditing(false); setStatus(null) }}>Cancel</button>
            <button className={styles.saveBtn} onClick={handleSave} disabled={status === 'saving'}>
              {status === 'saving' ? 'Saving…' : 'Save correction'}
            </button>
          </div>
          {status === 'error' && <span className={styles.errorMsg}>Failed to save. Try again.</span>}
        </div>
      ) : (
        <div className={styles.findingRow}>
          <span className={styles.finding}>{memory.result_summary}</span>
          {status === 'saved'
            ? <span className={styles.savedMsg}>✓ Corrected</span>
            : <button className={styles.correctBtn} onClick={() => setEditing(true)}>✏️ Correct</button>
          }
        </div>
      )}
    </div>
  )
}

export default function MemoryFeedback({ memories }) {
  if (!memories || memories.length === 0) return null
  return (
    <div className={styles.container}>
      <p className={styles.heading}>📚 Past cases recalled as context</p>
      {memories.map((m, i) => <MemoryEntry key={i} memory={m} />)}
    </div>
  )
}
