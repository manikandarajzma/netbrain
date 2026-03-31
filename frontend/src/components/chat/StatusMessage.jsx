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

export default function StatusMessage({ text, steps = [] }) {
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
      {steps.map((step, i) => (
        <div key={i} className={`${styles.completedStep} ${step.duration >= 5 ? styles.stepSlow : ''}`}>
          <span className={styles.checkMark}>✓</span>
          <span className={styles.completedLabel}>{step.label}</span>
          <span className={styles.stepDuration}>{step.duration.toFixed(1)}s</span>
        </div>
      ))}
      {displayText && (
        <div className={`${styles.currentStep} ${fading ? styles.statusFade : ''}`}>
          <span className={styles.currentLabel}>{displayText}</span>
          <span className={styles.typingDots}><span /><span /><span /></span>
        </div>
      )}
    </div>
  )
}
