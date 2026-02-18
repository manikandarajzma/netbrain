/**
 * Thin fetch wrappers for all API endpoints.
 * All requests include credentials (same-origin cookies).
 */

export async function fetchMe() {
  const res = await fetch('/api/me')
  if (!res.ok) throw new Error('Not authenticated')
  return res.json()
}

export async function fetchHealth() {
  const res = await fetch('/health', { signal: AbortSignal.timeout(5000) })
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}

export async function discoverTool(message, conversationHistory, signal) {
  const res = await fetch('/api/discover', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, conversation_history: conversationHistory }),
    signal,
  })
  if (!res.ok) throw new Error('Discover failed')
  return res.json()
}

export async function sendChat(message, conversationHistory, signal) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, conversation_history: conversationHistory }),
    signal,
  })
  if (!res.ok) throw new Error('Chat failed')
  return res.json()
}

export async function uploadBatch(file, message, signal) {
  const form = new FormData()
  form.append('file', file)
  form.append('message', message)
  const res = await fetch('/api/batch-upload', {
    method: 'POST',
    body: form,
    signal,
  })
  if (!res.ok) throw new Error('Upload failed')
  return res.json()
}
