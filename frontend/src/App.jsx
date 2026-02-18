import { useEffect } from 'react'
import { useUserStore } from './stores/userStore.js'
import ChatLayout from './components/layout/ChatLayout.jsx'
import BackgroundParticles from './components/particles/BackgroundParticles.jsx'
import styles from './App.module.css'

export default function App() {
  const loading = useUserStore(s => s.loading)
  const isAuthenticated = useUserStore(s => s.isAuthenticated)
  const loadUser = useUserStore(s => s.loadUser)

  useEffect(() => { loadUser() }, [loadUser])

  if (loading) {
    return <div className={styles.appLoading}>Loading...</div>
  }

  if (!isAuthenticated) return null

  return (
    <>
      <BackgroundParticles />
      <ChatLayout />
    </>
  )
}
