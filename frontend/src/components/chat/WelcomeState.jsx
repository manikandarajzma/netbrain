import styles from './WelcomeState.module.css'

export default function WelcomeState() {
  return (
    <div className={styles.welcomeState}>
      <div className={styles.welcomeTitle}>What can I help with?</div>
      <div className={styles.welcomeSubtitle}>Ask about network paths, device racks, Panorama groups, or Splunk logs.</div>
    </div>
  )
}
