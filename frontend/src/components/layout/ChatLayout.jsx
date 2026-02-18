import { useRef, useCallback } from 'react'
import AppHeader from './AppHeader.jsx'
import AppSidebar from './AppSidebar.jsx'
import ChatMessages from '../chat/ChatMessages.jsx'
import ChatInput from '../chat/ChatInput.jsx'
import styles from './ChatLayout.module.css'

export default function ChatLayout() {
  const inputRef = useRef(null)

  const fillInput = useCallback((query) => {
    if (inputRef.current) inputRef.current.fillText(query)
  }, [])

  return (
    <>
      <AppHeader />
      <div className={styles.appMain}>
        <AppSidebar onFillInput={fillInput} />
        <div className={styles.chatMain}>
          <ChatMessages />
          <ChatInput ref={inputRef} />
        </div>
      </div>
    </>
  )
}
