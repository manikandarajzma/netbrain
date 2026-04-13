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

  const groupedCounters = Array.from(
    counters.reduce((acc, entry) => {
      const deviceName = entry?.device || 'Unknown device'
      const current = acc.get(deviceName) || {
        device: deviceName,
        window_s: 0,
        ssh_error: '',
        activeByInterface: new Map(),
        cleanSet: new Set(),
      }

      current.window_s = Math.max(current.window_s || 0, entry?.window_s || 0)
      if (!current.ssh_error && entry?.ssh_error) {
        current.ssh_error = entry.ssh_error
      }

      for (const intf of entry?.active || []) {
        const key = intf?.interface || `active-${current.activeByInterface.size}`
        current.activeByInterface.set(key, intf)
      }

      for (const intf of entry?.clean || []) {
        if (intf) current.cleanSet.add(intf)
      }

      acc.set(deviceName, current)
      return acc
    }, new Map()).values()
  ).map((device) => {
    const active = Array.from(device.activeByInterface.values())
    const activeInterfaces = new Set(active.map((intf) => intf?.interface).filter(Boolean))
    const clean = Array.from(device.cleanSet).filter((intf) => !activeInterfaces.has(intf))
    return {
      device: device.device,
      window_s: device.window_s,
      ssh_error: device.ssh_error,
      active,
      clean,
    }
  })

  return (
    <div className={styles.container}>
      <p className={styles.heading}>Interface Error Counters</p>
      {groupedCounters.map((device, di) => {
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
