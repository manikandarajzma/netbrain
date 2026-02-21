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

export async function discoverTool(message, conversationHistory, signal) {
  const res = await fetch('/api/discover', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, conversation_history: conversationHistory }),
    signal,
  })
  checkAuthRedirect(res)
  if (!res.ok) throw new Error('Discover failed')
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

export async function sendChat(message, conversationHistory, signal, conversationId = null, parentConversationId = null) {
  const body = { message, conversation_history: conversationHistory }
  if (conversationId) body.conversation_id = conversationId
  if (parentConversationId) body.parent_conversation_id = parentConversationId
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  checkAuthRedirect(res)
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
  checkAuthRedirect(res)
  if (!res.ok) throw new Error('Upload failed')
  return res.json()
}
