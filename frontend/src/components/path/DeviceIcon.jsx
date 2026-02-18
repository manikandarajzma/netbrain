import { useState } from 'react'
import { deviceTypeToIcon, deviceTypeToFallbackLabel } from '../../utils/deviceIcons.js'
import styles from './DeviceIcon.module.css'

export default function DeviceIcon({ deviceType }) {
  const iconSrc = deviceTypeToIcon(deviceType)
  const fallbackLabel = deviceTypeToFallbackLabel(deviceType)
  const [imgFailed, setImgFailed] = useState(false)

  return (
    <div className={styles.pathIconWrap}>
      {iconSrc && !imgFailed ? (
        <img
          className={styles.pathIcon}
          src={iconSrc}
          alt={deviceType || ''}
          onError={() => setImgFailed(true)}
        />
      ) : (
        <span className={styles.pathIconFallback}>{fallbackLabel}</span>
      )}
    </div>
  )
}
