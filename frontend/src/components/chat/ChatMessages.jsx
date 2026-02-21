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
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, isLoading])

  return (
    <div className={styles.messages}>
      {messages.length === 0 && !isLoading && <WelcomeState />}
      {messages.map(msg => (
        <MessageBubble key={msg.id} role={msg.role} content={msg.content} />
      ))}
      {isLoading && <StatusMessage text={currentStatus} />}
      <div ref={bottomRef} />
    </div>
  )
}
