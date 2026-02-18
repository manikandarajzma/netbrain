import { useState, useEffect, useRef } from 'react'
import styles from './StatusMessage.module.css'

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

  return (
    <div className={styles.statusMsg}>
      <span className={`${styles.statusText} ${fading ? styles.statusFade : ''}`}>{displayText}</span>
      <span className={styles.typingDots}>
        <span /><span /><span />
      </span>
    </div>
  )
}
