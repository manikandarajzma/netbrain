import { useState, useCallback } from 'react'

function getInitialTheme() {
  const stored = localStorage.getItem('theme')
  const theme = stored || 'dark'
  document.documentElement.setAttribute('data-theme', theme)
  return theme
}

const initialTheme = getInitialTheme()

export function useTheme() {
  const [theme, setTheme] = useState(initialTheme)

  const toggleTheme = useCallback(() => {
    setTheme(prev => {
      const next = prev === 'dark' ? 'light' : 'dark'
      localStorage.setItem('theme', next)
      document.documentElement.setAttribute('data-theme', next)
      return next
    })
  }, [])

  return { theme, toggleTheme }
}
