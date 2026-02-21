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
  const startFollowUp = useChatStore(s => s.startFollowUp)
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
      <div className={styles.yourChats}>
        <h2 className={styles.heading}>Your chats</h2>
        <div className={styles.newChatRow}>
          <button type="button" className={styles.newChatBtn} onClick={newChat}>
            <span className={styles.newChatIcon}>+</span> New chat
          </button>
          <button type="button" className={styles.clearChatBtn} onClick={clearChat} title="Clear current chat and remove from list">
            Clear
          </button>
          {activeConversationId && (
            <button type="button" className={styles.followUpBtn} onClick={startFollowUp} title="Start a follow-up under this chat">
              Follow-up
            </button>
          )}
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
              <span className={styles.conversationTitle}>{c.title || 'New chat'}</span>
            </button>
          ))}
        </div>
        <button type="button" className={styles.clearAllChatsBtn} onClick={clearAllChats} title="Delete all past chats">
          Clear all past chats
        </button>
      </div>
    </aside>
  )
}
