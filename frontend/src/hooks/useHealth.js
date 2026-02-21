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
        const mcp = data.mcp_server
        const tools = data.mcp_tools_registered
        const ollama = data.ollama || {}
        const ollamaOk = ollama.status === 'ok'
        const ollamaUnreachable = ollama.status === 'unreachable'
        const ollamaModelMissing = ollama.status === 'model_not_found'

        if (mcp === 'ok' && tools > 0 && ollamaOk) {
          setStatus('healthy')
          setLabel('All systems OK')
          setTooltip(`MCP: OK | Tools: ${tools} | Ollama: ${ollama.model}`)
        } else if (mcp === 'unreachable') {
          setStatus('degraded')
          setLabel('MCP offline')
          setTooltip('MCP server is unreachable')
        } else if (ollamaUnreachable) {
          setStatus('degraded')
          setLabel('Ollama offline')
          setTooltip(`Ollama is not running (${ollama.url})`)
        } else if (ollamaModelMissing) {
          setStatus('degraded')
          setLabel('Model not found')
          setTooltip(`Ollama is running but model '${ollama.model}' is not pulled`)
        } else {
          setStatus('degraded')
          setLabel('System issue')
          setTooltip(`MCP: ${mcp} | Ollama: ${ollama.status || 'unknown'}`)
        }
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
