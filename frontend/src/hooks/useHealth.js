import { useState, useEffect, useRef } from 'react'
import { fetchHealth } from '../utils/api.js'

export function useHealth() {
  const [status, setStatus] = useState('checking')
  const [label, setLabel] = useState('Checking...')
  const [tooltip, setTooltip] = useState('Checking...')
  const timerRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const data = await fetchHealth()
        if (cancelled) return
        const overall = data.overall || {}
        const services = data.services || {}
        const mcp = services.mcp || {}
        const ollama = services.ollama || {}
        const nornir = services.nornir || {}

        setStatus(overall.status || 'degraded')
        setLabel(overall.label || 'System issue')
        setTooltip(
          `MCP: ${mcp.status || 'unknown'} | ` +
          `Ollama: ${ollama.status || 'unknown'} | ` +
          `Nornir: ${nornir.status || 'unknown'}`
        )
      } catch {
        if (cancelled) return
        setStatus('unhealthy')
        setLabel('Offline')
        setTooltip('Server unreachable')
      }
    }

    poll()
    timerRef.current = setInterval(poll, 30000)
    return () => {
      cancelled = true
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  return { status, label, tooltip }
}
