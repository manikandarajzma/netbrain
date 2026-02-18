import { useState, useEffect } from 'react'
import styles from './BackgroundParticles.module.css'

const colors = [
  'rgba(56, 139, 253, 0.06)',
  'rgba(139, 148, 158, 0.04)',
  'rgba(63, 185, 80, 0.04)',
]

export default function BackgroundParticles() {
  const [particles, setParticles] = useState([])

  useEffect(() => {
    const items = []
    for (let i = 0; i < 15; i++) {
      items.push({
        size: 3 + Math.random() * 5,
        left: Math.random() * 100,
        color: colors[Math.floor(Math.random() * colors.length)],
        duration: 15 + Math.random() * 25,
        delay: Math.random() * 20,
      })
    }
    setParticles(items)
  }, [])

  return (
    <div className={styles.container}>
      {particles.map((p, i) => (
        <div
          key={i}
          className={styles.particle}
          style={{
            width: p.size + 'px',
            height: p.size + 'px',
            left: p.left + '%',
            background: p.color,
            animationDuration: p.duration + 's',
            animationDelay: p.delay + 's',
          }}
        />
      ))}
    </div>
  )
}
