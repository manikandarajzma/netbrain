import { useEffect, useState } from 'react'
import { useUserStore } from './stores/userStore.js'
import ChatLayout from './components/layout/ChatLayout.jsx'
import BackgroundParticles from './components/particles/BackgroundParticles.jsx'
import Dashboard from './components/dashboard/Dashboard.jsx'
import AppHeader from './components/layout/AppHeader.jsx'
import styles from './App.module.css'

export default function App() {
  const loading = useUserStore(s => s.loading)
  const isAuthenticated = useUserStore(s => s.isAuthenticated)
  const loadUser = useUserStore(s => s.loadUser)
  const [view, setView] = useState('chat')

  useEffect(() => { loadUser() }, [loadUser])

  if (loading) {
    return <div className={styles.appLoading}>Loading...</div>
  }

  if (!isAuthenticated) return null

  return (
    <>
      <BackgroundParticles />
      <AppHeader view={view} onViewChange={setView} />
      <div className={styles.appMain}>
        {view === 'dashboard'
          ? <Dashboard />
          : <ChatLayout headerless />
        }
      </div>
    </>
  )
}
