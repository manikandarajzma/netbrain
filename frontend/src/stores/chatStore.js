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

export const useChatStore = create((set, get) => ({
  messages: [],
  conversationHistory: [],
  conversations: [],
  activeConversationId: null,
  nextConversationParentId: null,
  isLoading: false,
  currentStatus: '',
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

  addMessage: (role, content) => {
    set(s => ({ messages: [...s.messages, { id: nextId++, role, content }] }))
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
    addMessage('user', text)
    pushHistory('user', text)

    const ctrl = new AbortController()
    set({ isLoading: true, currentStatus: 'Identifying query', abortController: ctrl })
    const signal = ctrl.signal
    const historySlice = get().conversationHistory.slice(-20)
    const parentIdForNew = nextConversationParentId || null
    // Never append to current conversation: each send creates a new conversation so sidebar entries stay one Q&A each.
    const conversationIdToUse = null

    try {
      let toolDisplayName = null
      try {
        const discoverData = await discoverTool(text, historySlice, signal)
        toolDisplayName = discoverData.tool_display_name
        set({ currentStatus: toolDisplayName ? 'Querying ' + toolDisplayName : 'Processing' })
      } catch (err) {
        if (err && err.name === 'AbortError') throw err
        set({ currentStatus: 'Processing' })
      }

      const data = await sendChat(text, historySlice, signal, conversationIdToUse, parentIdForNew)

      if (toolDisplayName) {
        set({ currentStatus: 'Processing results from ' + toolDisplayName })
      }

      await new Promise(r => setTimeout(r, 400))

      const content = data.content ?? data.message ?? 'No response'
      addMessage('assistant', content)
      pushHistory('assistant', content)

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
      set({ isLoading: false, currentStatus: '', abortController: null })
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
      set({ isLoading: false, currentStatus: '', abortController: null })
    }
  },

  stopGeneration: () => {
    const { abortController } = get()
    if (abortController) abortController.abort()
    set({ abortController: null })
  },
}))
