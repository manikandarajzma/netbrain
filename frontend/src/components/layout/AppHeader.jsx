import { useUserStore } from '../../stores/userStore.js'
import { useTheme } from '../../hooks/useTheme.js'
import { useHealth } from '../../hooks/useHealth.js'
import styles from './AppHeader.module.css'

export default function AppHeader({ view, onViewChange }) {
  const displayName = useUserStore(s => s.displayName)
  const group = useUserStore(s => s.group)
  const { theme, toggleTheme } = useTheme()
  const { status, label, tooltip } = useHealth()

  return (
    <header className={styles.header}>
      <span className={styles.logo}>{'\u26A1'} Atlas</span>

      {/* View nav */}
      {onViewChange && (
        <nav className={styles.nav}>
          <button
            className={`${styles.navBtn} ${view === 'chat' ? styles.navBtnActive : ''}`}
            onClick={() => onViewChange('chat')}
          >
            Chat
          </button>
          <button
            className={`${styles.navBtn} ${view === 'dashboard' ? styles.navBtnActive : ''}`}
            onClick={() => onViewChange('dashboard')}
          >
            Dashboard
          </button>
        </nav>
      )}

      <div className={`${styles.healthStatus} ${styles[status] || ''}`} title={tooltip}>
        <span className={`${styles.healthDot} ${styles[status] || ''}`} />
        <span className={styles.healthLabel}>{label}</span>
      </div>
      <div className={styles.headerRight}>
        <span className={styles.userInfo}>
          {displayName()}
          <span className={styles.userRole}>{group}</span>
        </span>
        <button
          className={styles.themeToggle}
          title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
          aria-label="Toggle theme"
          onClick={toggleTheme}
        >
          <span className={styles.themeIcon}>{theme === 'light' ? '\u2600' : '\u263E'}</span>
        </button>
        <a href="/logout" className={styles.signOut}>Sign out {'\u2192'}</a>
      </div>
    </header>
  )
}
