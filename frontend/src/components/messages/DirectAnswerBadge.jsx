import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import styles from './DirectAnswerBadge.module.css'

export default function DirectAnswerBadge({ text }) {
  return (
    <div className={styles.directAnswer}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  )
}
