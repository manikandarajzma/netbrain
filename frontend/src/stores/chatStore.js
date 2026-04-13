import { create } from 'zustand'
import {
  discoverTool,
  sendChat,
  uploadBatch,
  fetchChatHistory,
  fetchConversations,
  fetchConversation,
  deleteConversation as apiDeleteConversation,
  clearChatHistory,
} from '../utils/api.js'

let nextId = 1
const nowMs = () => (typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now())
const stepSeconds = (start, end = nowMs()) => Math.max((end - start) / 1000, 0.001)

export const useChatStore = create((set, get) => ({
  messages: [],
  conversationHistory: [],
  conversations: [],
  activeConversationId: null,
  nextConversationParentId: null,
  isLoading: false,
  currentStatus: '',
  statusSteps: [],
  _stepStart: null,
  abortController: null,
  historyLoaded: false,

  loadHistory: async () => {
    try {
      const { messages: saved } = await fetchChatHistory()
      if (!Array.isArray(saved) || saved.length === 0) {
        set({ messages: [], conversationHistory: [], historyLoaded: true })
        return
      }
      const messages = saved.map((m) => ({ id: nextId++, role: m.role, content: m.content }))
      const conversationHistory = saved.map(m => ({
        role: m.role,
        content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content),
      }))
      set({ messages, conversationHistory, historyLoaded: true })
    } catch {
      set({ historyLoaded: true })
    }
  },

  loadConversations: async () => {
    try {
      const conversations = await fetchConversations()
      set({ conversations: Array.isArray(conversations) ? conversations : [] })
    } catch (_) {
      set({ conversations: [] })
    }
  },

  selectConversation: async (id) => {
    if (!id) {
      set({ messages: [], conversationHistory: [], activeConversationId: null })
      return
    }
    try {
      const saved = await fetchConversation(id)
      const messages = (Array.isArray(saved) ? saved : []).map((m) => ({
        id: nextId++,
        role: m.role,
        content: m.content,
      }))
      const conversationHistory = messages.map(m => ({
        role: m.role,
        content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content),
      }))
      set({ messages, conversationHistory, activeConversationId: id })
    } catch (_) {
      set({ messages: [], conversationHistory: [], activeConversationId: id })
    }
  },

  newChat: () => {
    set({ messages: [], conversationHistory: [], activeConversationId: null, nextConversationParentId: null })
  },

  clearChat: async () => {
    const { activeConversationId, deleteConversation, newChat } = get()
    if (activeConversationId) {
      await deleteConversation(activeConversationId)
    }
    newChat()
  },

  startFollowUp: () => {
    const id = get().activeConversationId
    if (id) set({ nextConversationParentId: id })
  },

  deleteConversation: async (id) => {
    try {
      await apiDeleteConversation(id)
      const { activeConversationId, conversations } = get()
      const next = conversations.filter((c) => c.id !== id)
      set({ conversations: next })
      if (activeConversationId === id) {
        set({ messages: [], conversationHistory: [], activeConversationId: null })
      }
    } catch (_) {}
  },

  clearHistory: async () => {
    try {
      await clearChatHistory()
      set({ messages: [], conversationHistory: [], conversations: [], activeConversationId: null })
    } catch (_) {}
  },

  addMessage: (role, content, memories = null) => {
    set(s => ({ messages: [...s.messages, { id: nextId++, role, content, memories: memories || [] }] }))
  },

  pushHistory: (role, content) => {
    const text = typeof content === 'string' ? content : JSON.stringify(content)
    set(s => {
      const h = [...s.conversationHistory, { role, content: text }]
      return { conversationHistory: h.length > 20 ? h.slice(-20) : h }
    })
  },

  sendMessage: async (text) => {
    if (!text.trim()) return

    const { addMessage, pushHistory, nextConversationParentId } = get()

    // If the last assistant message was a requires_site clarification, combine
    // the user's site reply into a full query so the LLM gets complete context
    // in a single message without needing conversation history.
    let textToSend = text
    const history = get().conversationHistory
    const lastAssistant = [...history].reverse().find(m => m.role === 'assistant')
    if (lastAssistant) {
      let lc = lastAssistant.content
      if (typeof lc === 'string' && lc.startsWith('{')) { try { lc = JSON.parse(lc) } catch (_) {} }
      if (lc && typeof lc === 'object' && lc.requires_site && lc.rack) {
        textToSend = `${lc.rack} at ${text.trim()}`
      }
    }

    addMessage('user', text)
    pushHistory('user', text)

    const ctrl = new AbortController()
    set({ isLoading: true, currentStatus: 'Identifying query', statusSteps: [], _stepStart: nowMs(), abortController: ctrl })
    const signal = ctrl.signal
    const historySlice = get().conversationHistory
    const parentIdForNew = nextConversationParentId || null
    // Never append to current conversation: each send creates a new conversation so sidebar entries stay one Q&A each.
    const conversationIdToUse = null

    try {
      // Start both calls simultaneously — discover only drives a neutral routing label
      let toolDisplayName = null
      const discoverPromise = discoverTool(textToSend, historySlice, signal)
        .then(d => {
          toolDisplayName = d.tool_display_name
          const newStatus = 'Routing request'
          const now = nowMs()
          const { currentStatus, statusSteps, _stepStart } = get()
          if (currentStatus && _stepStart) {
            set({
              statusSteps: [...statusSteps, { label: currentStatus, duration: stepSeconds(_stepStart, now) }],
              currentStatus: newStatus,
              _stepStart: now,
            })
          } else {
            set({ currentStatus: newStatus, _stepStart: now })
          }
        })
        .catch(err => {
          if (err && err.name === 'AbortError') throw err
          set({ currentStatus: 'Routing request' })
        })

      const data = await sendChat(textToSend, historySlice, signal, conversationIdToUse, parentIdForNew, (msg) => {
        const now = nowMs()
        const { currentStatus, statusSteps, _stepStart } = get()
        if (currentStatus && _stepStart) {
          set({
            statusSteps: [...statusSteps, { label: currentStatus, duration: stepSeconds(_stepStart, now) }],
            currentStatus: msg,
            _stepStart: now,
          })
        } else {
          set({ currentStatus: msg, _stepStart: now })
        }
      })
      await discoverPromise

      if (toolDisplayName) {
        const now = nowMs()
        const { currentStatus, statusSteps, _stepStart } = get()
        if (currentStatus && _stepStart) {
          set({
            statusSteps: [...statusSteps, { label: currentStatus, duration: stepSeconds(_stepStart, now) }],
            currentStatus: 'Preparing response',
            _stepStart: now,
          })
        } else {
          set({ currentStatus: 'Preparing response', _stepStart: now })
        }
      }

      await new Promise(r => setTimeout(r, 400))

      // If the response includes structured path hops, wrap them together with
      // the markdown text so the PathVisualization component can render the
      // diagram while the analysis text is still shown below it.
      const rawContent = data.content ?? data.message ?? 'No response'
      const content = data.path_hops?.length
        ? { text: rawContent, path_hops: data.path_hops, ...(data.reverse_path_hops?.length ? { reverse_path_hops: data.reverse_path_hops } : {}) }
        : rawContent
      const memories = Array.isArray(data.memories) ? data.memories : []
      addMessage('assistant', content, memories)
      pushHistory('assistant', rawContent)

      if (data.conversation_id) {
        set({ activeConversationId: data.conversation_id, nextConversationParentId: null })
        get().loadConversations()
      }
    } catch (err) {
      if (err && err.name === 'AbortError') {
        addMessage('assistant', 'Request stopped.')
        pushHistory('assistant', 'Request stopped.')
      } else {
        let errMsg = err && err.message ? err.message : String(err)
        if (errMsg.includes('fetch')) errMsg = 'Request failed. Check that the server is running.'
        addMessage('assistant', 'Error: ' + errMsg)
        pushHistory('assistant', 'Error: ' + errMsg)
      }
    } finally {
      // Push the last in-progress step as completed before clearing
      const { currentStatus: lastStatus, statusSteps: lastSteps, _stepStart: lastStart } = get()
      const finalSteps = lastStatus && lastStart
        ? [...lastSteps, { label: lastStatus, duration: stepSeconds(lastStart) }]
        : lastSteps
      set({ isLoading: false, currentStatus: '', statusSteps: finalSteps, _stepStart: null, abortController: null })
    }
  },

  sendBatch: async (file, message) => {
    const { addMessage } = get()
    const msg = message || 'check if paths are allowed'
    addMessage('user', msg + '\n\uD83D\uDCCE ' + file.name)

    const ctrl = new AbortController()
    set({ isLoading: true, currentStatus: 'Processing batch upload', abortController: ctrl })

    try {
      const data = await uploadBatch(file, msg, ctrl.signal)
      const content = data.content || data
      addMessage('assistant', content)
    } catch (err) {
      if (err && err.name === 'AbortError') {
        addMessage('assistant', 'Request stopped.')
      } else {
        const errMsg = err && err.message ? err.message : String(err)
        addMessage('assistant', 'Upload error: ' + errMsg)
      }
    } finally {
      set({ isLoading: false, currentStatus: '', statusSteps: [], _stepStart: null, abortController: null })
    }
  },

  stopGeneration: () => {
    const { abortController } = get()
    if (abortController) abortController.abort()
    set({ abortController: null })
  },
}))
