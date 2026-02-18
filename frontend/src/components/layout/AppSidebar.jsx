import { useState } from 'react'
import { useUserStore } from '../../stores/userStore.js'
import { exampleQueries } from '../../utils/exampleQueries.js'
import styles from './AppSidebar.module.css'

export default function AppSidebar({ onFillInput }) {
  const allowedCategories = useUserStore(s => s.allowedCategories)
  const [collapsed, setCollapsed] = useState(() => {
    const init = {}
    exampleQueries.forEach(cat => { init[cat.category] = true })
    return init
  })

  function toggleCategory(cat) {
    setCollapsed(prev => ({ ...prev, [cat]: !prev[cat] }))
  }

  function isCategoryAllowed(category) {
    return allowedCategories === null || allowedCategories.includes(category)
  }

  return (
    <aside className={styles.sidebar}>
      <div className={styles.exampleQueries}>
        <h2 className={styles.heading}>{'\uD83D\uDCCB'} Example Queries</h2>
        {exampleQueries.map(cat => (
          isCategoryAllowed(cat.category) && (
            <div
              key={cat.category}
              className={`${styles.category} ${collapsed[cat.category] ? styles.collapsed : ''}`}
            >
              <div className={styles.categoryLabel} onClick={() => toggleCategory(cat.category)}>
                {cat.icon} {cat.label}
              </div>
              <div className={styles.chips}>
                {cat.queries.map(q => (
                  <button
                    key={q}
                    type="button"
                    className={styles.chip}
                    onClick={() => onFillInput(q)}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )
        ))}
      </div>
    </aside>
  )
}
