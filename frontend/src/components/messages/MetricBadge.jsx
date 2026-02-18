import styles from './MetricBadge.module.css'

export default function MetricBadge({ text }) {
  return <div className={styles.metricAnswer}>{text}</div>
}
