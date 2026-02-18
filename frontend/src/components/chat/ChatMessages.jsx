import { useRef, useEffect } from 'react'
import { useChatStore } from '../../stores/chatStore.js'
import WelcomeState from './WelcomeState.jsx'
import StatusMessage from './StatusMessage.jsx'
import MessageBubble from '../messages/MessageBubble.jsx'
import styles from './ChatMessages.module.css'

export default function ChatMessages() {
  const messages = useChatStore(s => s.messages)
  const isLoading = useChatStore(s => s.isLoading)
  const currentStatus = useChatStore(s => s.currentStatus)
  const messagesRef = useRef(null)

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight
    }
  }, [messages.length, isLoading])

  return (
    <div className={styles.messages} ref={messagesRef}>
      {messages.length === 0 && !isLoading && <WelcomeState />}
      {messages.map(msg => (
        <MessageBubble key={msg.id} role={msg.role} content={msg.content} />
      ))}
      {isLoading && <StatusMessage text={currentStatus} />}
    </div>
  )
}
