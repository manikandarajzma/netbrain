import ReactMarkdown from 'react-markdown'
import styles from './MarkdownContent.module.css'

// Links in documentation point to local .md file paths which have no web URL.
// Render the link text as a styled inline code span instead of a broken anchor.
function DocLink({ children }) {
  return <code className={styles.docRef}>{children}</code>
}

export default function MarkdownContent({ text }) {
  return (
    <div className={styles.root}>
      <ReactMarkdown components={{ a: DocLink }}>{text}</ReactMarkdown>
    </div>
  )
}
