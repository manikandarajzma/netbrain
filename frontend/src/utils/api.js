/**
 * Thin fetch wrappers for all API endpoints.
 * All requests include credentials (same-origin cookies).
 * On 401 we redirect to /login so the user can sign in again.
 */

function checkAuthRedirect(res) {
  if (res.status === 401) {
    window.location.href = '/login'
    throw new Error('Not authenticated')
  }
}

export async function fetchMe() {
  const res = await fetch('/api/me')
  checkAuthRedirect(res)
  if (!res.ok) throw new Error('Not authenticated')
  return res.json()
}

export async function fetchHealth() {
  const res = await fetch('/health', { signal: AbortSignal.timeout(5000) })
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}

export async function fetchDiagnostics() {
  const res = await fetch('/api/internal/diagnostics', { signal: AbortSignal.timeout(5000) })
  checkAuthRedirect(res)
  if (!res.ok) throw new Error('Failed to load diagnostics')
  return res.json()
}

export async function fetchChatHistory() {
  const res = await fetch('/api/chat/history')
  checkAuthRedirect(res)
  if (!res.ok) throw new Error('Failed to load chat history')
  return res.json()
}

export async function fetchConversations() {
  const res = await fetch('/api/chat/conversations')
  checkAuthRedirect(res)
  if (!res.ok) throw new Error('Failed to load conversations')
  const data = await res.json()
  return data.conversations ?? []
}

export async function fetchConversation(id) {
  const res = await fetch(`/api/chat/conversations/${encodeURIComponent(id)}`)
  checkAuthRedirect(res)
  if (!res.ok) throw new Error('Failed to load conversation')
  const data = await res.json()
  return data.messages ?? []
}

export async function deleteConversation(id) {
  const res = await fetch(`/api/chat/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' })
  checkAuthRedirect(res)
  if (!res.ok) throw new Error('Failed to delete conversation')
  return res.json()
}

export async function clearChatHistory() {
  const res = await fetch('/api/chat/history', { method: 'DELETE' })
  checkAuthRedirect(res)
  if (!res.ok) throw new Error('Failed to clear chat history')
  return res.json()
}

// Chat can take a long time (NetBrain path + troubleshoot orchestrator can take up to ~10 minutes).
const CHAT_FETCH_TIMEOUT_MS = 600000

export async function sendChat(message, conversationHistory, signal, conversationId = null, parentConversationId = null, onStatus = null, uiAction = null) {
  const body = { message, conversation_history: conversationHistory }
  if (conversationId) body.conversation_id = conversationId
  if (parentConversationId) body.parent_conversation_id = parentConversationId
  if (uiAction) body.ui_action = uiAction
  const timeoutSignal = AbortSignal.timeout ? AbortSignal.timeout(CHAT_FETCH_TIMEOUT_MS) : null
  const combinedSignal = timeoutSignal && signal ? (AbortSignal.any ? AbortSignal.any([signal, timeoutSignal]) : signal) : (timeoutSignal || signal)
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: combinedSignal,
  })
  checkAuthRedirect(res)
  if (!res.ok) {
    let msg = 'Chat failed'
    try {
      const err = await res.json()
      const raw = (err?.detail && typeof err.detail === 'string' ? err.detail : null) || err?.error
      if (raw && typeof raw === 'string' && !/keyvault|api-version|REDACTED|\.py\s|traceback/i.test(raw)) {
        msg = raw.length > 200 ? raw.slice(0, 200) + '…' : raw
      }
    } catch (_) {}
    throw new Error(msg)
  }

  // Response is SSE — read the stream until we get a "done" event
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() // keep incomplete last line
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      try {
        const event = JSON.parse(line.slice(6))
        if (event.type === 'status' && onStatus) {
          onStatus(event.message)
          await new Promise(r => setTimeout(r, 0)) // yield to event loop so React re-renders each step
        } else if (event.type === 'done') {
          return event.result
        }
      } catch (_) {}
    }
  }
  throw new Error('Chat stream ended without a response')
}


export async function fetchTopology() {
  const res = await fetch('/api/topology')
  checkAuthRedirect(res)
  if (!res.ok) throw new Error('Failed to load topology')
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
  checkAuthRedirect(res)
  if (!res.ok) throw new Error('Upload failed')
  return res.json()
}
