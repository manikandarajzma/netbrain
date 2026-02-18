import { useState, useEffect, useRef } from 'react'
import { fetchHealth } from '../utils/api.js'

export function useHealth() {
  const [status, setStatus] = useState('checking')
  const [label, setLabel] = useState('Checking...')
  const [tooltip, setTooltip] = useState('Checking...')
  const timerRef = useRef(null)

  useEffect(() => {
    async function poll() {
      try {
        const data = await fetchHealth()
        const mcp = data.mcp_server
        const tools = data.mcp_tools_registered

        if (mcp === 'ok' && tools > 0) {
          setStatus('healthy')
          setLabel('All systems OK')
          setTooltip(`MCP: OK | Tools: ${tools}`)
        } else if (mcp === 'unreachable') {
          setStatus('degraded')
          setLabel('MCP offline')
          setTooltip('MCP server is unreachable')
        } else {
          setStatus('degraded')
          setLabel('MCP issue')
          setTooltip(`MCP: ${mcp}`)
        }
      } catch {
        setStatus('unhealthy')
        setLabel('Offline')
        setTooltip('Server unreachable')
      }
    }

    poll()
    timerRef.current = setInterval(poll, 30000)
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [])

  return { status, label, tooltip }
}
