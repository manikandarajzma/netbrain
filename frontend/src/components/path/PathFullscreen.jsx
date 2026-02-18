import { useEffect } from 'react'
import { createPortal } from 'react-dom'
import styles from './PathFullscreen.module.css'

export default function PathFullscreen({ onClose, children }) {
  useEffect(() => {
    function onEscape(e) {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onEscape)
    return () => document.removeEventListener('keydown', onEscape)
  }, [onClose])

  function onBackdropClick(e) {
    if (e.target.classList.contains(styles.overlay) || e.target.classList.contains(styles.body)) {
      onClose()
    }
  }

  return createPortal(
    <div className={styles.overlay} onClick={onBackdropClick}>
      <div className={styles.header}>
        <span className={styles.title}>Network Path</span>
        <button className={styles.closeBtn} onClick={onClose}>Close (Esc)</button>
      </div>
      <div className={styles.body}>
        {children}
      </div>
    </div>,
    document.body
  )
}
