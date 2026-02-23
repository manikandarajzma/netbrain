import { create } from 'zustand'
import { fetchMe } from '../utils/api.js'

export const useUserStore = create((set) => ({
  username: '',
  group: '',
  allowedCategories: null,
  isAuthenticated: false,
  loading: true,

  loadUser: async () => {
    set({ loading: true })
    try {
      const data = await fetchMe()
      set({
        username: data.username || '',
        group: data.group || '',
        allowedCategories: data.allowed_categories,
        isAuthenticated: true,
        loading: false,
      })
    } catch {
      set({ isAuthenticated: false, loading: false })
      window.location.href = '/login'
    }
  },

  displayName: () => {
    const { username } = useUserStore.getState()
    return username.split('@')[0]
  },
}))
