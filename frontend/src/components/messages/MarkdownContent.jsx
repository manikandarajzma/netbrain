import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import styles from './MarkdownContent.module.css'

// Links in documentation point to local .md file paths which have no web URL.
// Render the link text as a styled inline code span instead of a broken anchor.
function DocLink({ children }) {
  return <code className={styles.docRef}>{children}</code>
}

function TableWrapper({ children }) {
  return <div className={styles.tableWrapper}><table>{children}</table></div>
}

export default function MarkdownContent({ text }) {
  return (
    <div className={styles.root}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ a: DocLink, table: TableWrapper }}>{text}</ReactMarkdown>
    </div>
  )
}
