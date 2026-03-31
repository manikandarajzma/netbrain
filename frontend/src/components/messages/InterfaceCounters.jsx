import styles from './InterfaceCounters.module.css'

const COUNTER_LABELS = {
  inErrors:    'In Errors',
  outErrors:   'Out Errors',
  inDiscards:  'In Discards',
  outDiscards: 'Out Discards',
  fcsErrors:   'FCS Errors',
  runtFrames:  'Runts',
  giantFrames: 'Giants',
}

export default function InterfaceCounters({ counters }) {
  if (!counters || counters.length === 0) return null

  return (
    <div className={styles.container}>
      <p className={styles.heading}>Interface Error Counters</p>
      {counters.map((device, di) => {
        const active = device.active || []
        const clean  = device.clean  || []
        const window = device.window_s || 6
        return (
          <div key={di} className={styles.device}>
            <div className={styles.deviceHeader}>
              <span className={styles.deviceName}>{device.device}</span>
              <span className={styles.window}>{window}s window</span>
            </div>

            {device.ssh_error && (
              <p className={styles.sshError}>SSH unreachable — {device.ssh_error}</p>
            )}
            {!device.ssh_error && active.length === 0 && (
              <p className={styles.allClean}>All interfaces clean</p>
            )}

            {active.map((intf, ii) => {
              if (intf.error) return (
                <div key={ii} className={styles.intfRow}>
                  <span className={styles.intfName}>{intf.interface}</span>
                  <span className={styles.errorText}>{intf.error}</span>
                </div>
              )
              const deltas = intf.delta_9s || {}
              const active_counters = Object.entries(deltas).filter(([, v]) => v > 0)
              return (
                <div key={ii} className={styles.intfRow}>
                  <div className={styles.intfMeta}>
                    <span className={styles.intfName}>{intf.interface}</span>
                    <span className={styles.activeBadge}>ACTIVE</span>
                  </div>
                  <div className={styles.counters}>
                    {active_counters.map(([key, val]) => (
                      <span key={key} className={styles.counter}>
                        <span className={styles.counterLabel}>{COUNTER_LABELS[key] || key}</span>
                        <span className={styles.counterValue}>+{val}</span>
                      </span>
                    ))}
                  </div>
                  {intf.last_clear && intf.last_clear !== 'never' && (
                    <p className={styles.lastClear}>Cleared: {intf.last_clear}</p>
                  )}
                </div>
              )
            })}

            {clean.length > 0 && (
              <p className={styles.cleanList}>
                <span className={styles.cleanLabel}>Clean: </span>
                {clean.join(', ')}
              </p>
            )}
          </div>
        )
      })}
    </div>
  )
}
