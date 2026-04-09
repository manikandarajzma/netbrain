import { useRef, useCallback, useEffect } from 'react'
import { useChatStore } from '../../stores/chatStore.js'
import AppSidebar from './AppSidebar.jsx'
import ChatMessages from '../chat/ChatMessages.jsx'
import ChatInput from '../chat/ChatInput.jsx'
import styles from './ChatLayout.module.css'

const LAST_CHAT_KEY = 'atlas_last_conversation_id'

export default function ChatLayout() {
  const inputRef = useRef(null)
  const loadConversations = useChatStore(s => s.loadConversations)
  const selectConversation = useChatStore(s => s.selectConversation)
  const stopGeneration = useChatStore(s => s.stopGeneration)
  const activeConversationId = useChatStore(s => s.activeConversationId)

  useEffect(() => {
    loadConversations()
    const lastId = typeof localStorage !== 'undefined' ? localStorage.getItem(LAST_CHAT_KEY) : null
    if (lastId) selectConversation(lastId)
  }, [loadConversations, selectConversation])

  useEffect(() => {
    return () => { stopGeneration() }
  }, [stopGeneration])

  useEffect(() => {
    if (activeConversationId && typeof localStorage !== 'undefined') {
      localStorage.setItem(LAST_CHAT_KEY, activeConversationId)
    }
  }, [activeConversationId])

  const fillInput = useCallback((query) => {
    if (inputRef.current) inputRef.current.fillText(query)
  }, [])

  return (
    <div className={styles.appMainInner}>
      <AppSidebar onFillInput={fillInput} />
      <div className={styles.chatMain}>
        <div className={styles.chatScroll}>
          <ChatMessages />
        </div>
        <ChatInput ref={inputRef} />
      </div>
    </div>
  )
}
