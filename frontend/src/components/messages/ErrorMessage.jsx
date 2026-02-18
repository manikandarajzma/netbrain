import styles from './ErrorMessage.module.css'

export default function ErrorMessage({ content }) {
  return (
    <div>
      <p className={styles.errorText}>{content.error}</p>
    </div>
  )
}
