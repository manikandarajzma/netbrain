import { useMemo } from 'react'
import styles from './JsonFallback.module.css'

export default function JsonFallback({ content }) {
  const formatted = useMemo(() => {
    if (typeof content === 'string') return content
    return JSON.stringify(content, null, 2)
  }, [content])

  return <pre className={styles.pre}>{formatted}</pre>
}
