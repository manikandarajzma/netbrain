import styles from './DirectAnswerBadge.module.css'

export default function DirectAnswerBadge({ text }) {
  return <div className={styles.directAnswer}>{text}</div>
}
