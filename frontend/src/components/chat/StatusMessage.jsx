import { useState, useEffect, useRef } from 'react'
import styles from './StatusMessage.module.css'

function getPhase(text) {
  if (!text) return { label: 'Thinking', icon: '◈' }
  const t = text.toLowerCase()
  if (t.includes('identify') || t.includes('routing') || t.includes('classif')) return { label: 'Routing', icon: '⟁' }
  if (t.includes('querying') || t.includes('fetching') || t.includes('calling')) return { label: 'Querying', icon: '⬡' }
  if (t.includes('processing') || t.includes('synthesiz') || t.includes('analyz')) return { label: 'Processing', icon: '◎' }
  if (t.includes('path') || t.includes('netbrain')) return { label: 'Tracing path', icon: '⟁' }
  if (t.includes('panorama') || t.includes('firewall')) return { label: 'Querying', icon: '⬡' }
  if (t.includes('splunk')) return { label: 'Querying', icon: '⬡' }
  return { label: 'Working', icon: '◈' }
}

export default function StatusMessage({ text }) {
  const [displayText, setDisplayText] = useState(text)
  const [fading, setFading] = useState(false)
  const prevText = useRef(text)

  useEffect(() => {
    if (text !== prevText.current) {
      setFading(true)
      const timer = setTimeout(() => {
        setDisplayText(text)
        setFading(false)
        prevText.current = text
      }, 200)
      return () => clearTimeout(timer)
    }
  }, [text])

  const phase = getPhase(displayText)

  return (
    <div className={styles.bubble}>
      <div className={styles.header}>
        <span className={styles.phaseIcon}>{phase.icon}</span>
        <span className={styles.phaseLabel}>{phase.label}</span>
        <span className={styles.typingDots}>
          <span /><span /><span />
        </span>
      </div>
      {displayText && displayText.toLowerCase() !== phase.label.toLowerCase() && (
        <p className={`${styles.statusText} ${fading ? styles.statusFade : ''}`}>
          {displayText}
        </p>
      )}
    </div>
  )
}
