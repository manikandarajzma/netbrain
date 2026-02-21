import { useState } from 'react'
import { useUserStore } from '../../stores/userStore.js'
import { useChatStore } from '../../stores/chatStore.js'
import { exampleQueries } from '../../utils/exampleQueries.js'
import styles from './AppSidebar.module.css'

export default function AppSidebar({ onFillInput }) {
  const allowedCategories = useUserStore(s => s.allowedCategories)
  const conversations = useChatStore(s => s.conversations)
  const activeConversationId = useChatStore(s => s.activeConversationId)
  const newChat = useChatStore(s => s.newChat)
  const clearChat = useChatStore(s => s.clearChat)
  const clearAllChats = useChatStore(s => s.clearHistory)
  const selectConversation = useChatStore(s => s.selectConversation)
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

      {/* New Chat */}
      <div className={styles.navSection}>
        <button type="button" className={styles.newChatNavBtn} onClick={newChat}>
          <span className={styles.newChatNavIcon}>✎</span>
          New Chat
        </button>
      </div>

      {/* Example Queries */}
      <div className={styles.exampleQueries}>
        <p className={styles.sectionLabel}>📋 Example Queries</p>
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

      {/* History */}
      <div className={styles.history}>
        <div className={styles.historyHeader}>
          <p className={styles.sectionLabel}>History</p>
        </div>

        <div className={styles.conversationList}>
          {(conversations || []).map((c) => (
            <button
              key={c.id}
              type="button"
              className={`${styles.conversationItem} ${c.id === activeConversationId ? styles.active : ''} ${c.parent_id ? styles.child : ''}`}
              onClick={() => selectConversation(c.id)}
              title={c.title}
            >
              {c.title || 'New chat'}
            </button>
          ))}
        </div>

        <div className={styles.historyFooter}>
          <button type="button" className={styles.footerBtn} onClick={clearChat}>
            Clear current chat
          </button>
          <button type="button" className={`${styles.footerBtn} ${styles.footerBtnDanger}`} onClick={clearAllChats}>
            Clear all history
          </button>
        </div>
      </div>

    </aside>
  )
}
