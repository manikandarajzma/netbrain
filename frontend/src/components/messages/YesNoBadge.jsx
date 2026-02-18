import styles from './YesNoBadge.module.css'

export default function YesNoBadge({ text }) {
  return <div className={styles.yesNoAnswer}>{text}</div>
}
