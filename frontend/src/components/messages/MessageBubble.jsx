import AssistantMessage from './AssistantMessage.jsx'
import styles from './MessageBubble.module.css'

export default function MessageBubble({ role, content, memories }) {
  return (
    <div className={`${styles.msg} ${styles[role] || ''}`}>
      {role === 'user'
        ? (typeof content === 'string' ? content : JSON.stringify(content))
        : <AssistantMessage content={content} memories={memories || []} />
      }
    </div>
  )
}
