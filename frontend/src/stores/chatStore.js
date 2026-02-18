import { create } from 'zustand'
import { discoverTool, sendChat, uploadBatch } from '../utils/api.js'

let nextId = 1

export const useChatStore = create((set, get) => ({
  messages: [],
  conversationHistory: [],
  isLoading: false,
  currentStatus: '',
  abortController: null,

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

    const { addMessage, pushHistory } = get()
    addMessage('user', text)
    pushHistory('user', text)

    const ctrl = new AbortController()
    set({ isLoading: true, currentStatus: 'Identifying query', abortController: ctrl })
    const signal = ctrl.signal
    const historySlice = get().conversationHistory.slice(-20)

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

      const data = await sendChat(text, historySlice, signal)

      if (toolDisplayName) {
        set({ currentStatus: 'Processing results from ' + toolDisplayName })
      }

      await new Promise(r => setTimeout(r, 400))

      const content = data.content ?? data.message ?? 'No response'
      addMessage('assistant', content)
      pushHistory('assistant', content)
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
