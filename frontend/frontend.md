# Frontend — React 18 + Vite

## Prerequisites

- **Node.js** (v18+) installed and on PATH
- **FastAPI backend** running on port 8000

## Install Dependencies

```bash
cd atlas/frontend
npm install
```

## Development (Hot Reload)

Run two terminals:

**Terminal 1 — FastAPI backend:**
```bash
uvicorn atlas.app:app --reload --port 8000
```

**Terminal 2 — Vite dev server:**
```bash
cd atlas/frontend
npm run dev
```

- Use the app at `http://localhost:5173`
- Login at `http://localhost:8000/login` first (session cookie is shared)
- Vite proxies `/api/*`, `/health`, `/login`, `/logout`, `/auth/*`, `/icons/*`, `/static/*` to port 8000

## Production Build

```bash
cd atlas/frontend
npm run build
```

This outputs to `frontend/dist/`. FastAPI automatically serves the build when `frontend/dist/index.html` exists. If the dist folder is missing, it falls back to the old Jinja2 template.

Start the server as usual:
```bash
uvicorn atlas.app:app --port 8000
```

Visit `http://localhost:8000` — it serves the React app directly.

## Project Structure

```
frontend/
  package.json
  vite.config.js
  index.html                          # Vite entry (minimal shell)
  src/
    main.jsx                          # ReactDOM.createRoot, global CSS imports
    App.jsx                           # Root: fetches /api/me, renders ChatLayout
    App.module.css
    assets/
      variables.css                   # CSS custom properties (:root, light theme)
      base.css                        # Body, scrollbar, animations, fonts
    stores/
      userStore.js                    # Zustand: username, role, allowedCategories
      chatStore.js                    # Zustand: messages, history, sendMessage, stopGeneration
    hooks/
      useTheme.js                     # Custom hook: theme toggle (localStorage + data-theme)
      useHealth.js                    # Custom hook: health polling every 30s
    components/
      layout/
        AppHeader.jsx + .module.css   # Logo, health dot, user info, theme toggle, logout
        AppSidebar.jsx + .module.css  # Example queries by category, collapsible
        ChatLayout.jsx + .module.css  # Flexbox: header + sidebar + chat area
      chat/
        ChatMessages.jsx + .module.css  # Message list, auto-scroll, welcome state
        WelcomeState.jsx + .module.css  # Initial greeting with fade animation
        ChatInput.jsx + .module.css     # Text input, file upload, send/stop toggle
        StatusMessage.jsx + .module.css # Routing / progress badges with typing dots
      messages/
        MessageBubble.jsx + .module.css     # User vs assistant styling wrapper
        AssistantMessage.jsx + .module.css  # Routes content to correct renderer
        YesNoBadge.jsx + .module.css
        MetricBadge.jsx + .module.css
        DirectAnswerBadge.jsx + .module.css
        ErrorMessage.jsx + .module.css
        JsonFallback.jsx + .module.css
      path/
        PathVisualization.jsx + .module.css # Path view orchestrator
        PathItem.jsx + .module.css          # Single device node
        PathConnectors.jsx + .module.css    # SVG arrow lines between devices
        DeviceIcon.jsx + .module.css        # Device icon with fallback
        PathFullscreen.jsx + .module.css    # Portal fullscreen overlay
        FirewallDetails.jsx + .module.css   # Firewall details table + CSV export
      tables/
        DataTable.jsx + .module.css     # Generic table: filter, pagination, CSV export
        VerticalTable.jsx + .module.css # Key-value layout for single rows
        BatchResults.jsx + .module.css  # Summary stats + results table + path viz
      particles/
        BackgroundParticles.jsx + .module.css  # Floating particles (dark mode only)
    utils/
      api.js                          # Fetch wrappers: discoverTool, sendChat, etc.
      csvExport.js                    # CSV escape + blob download
      deviceIcons.js                  # Device type to icon mapping
      formatters.js                   # cellText, normalizeInterface, etc.
      responseClassifier.js           # Classifies response content by type
      exampleQueries.js               # Static query data by category
```

## Key Architecture Decisions

- **React 18** with functional components and hooks
- **Zustand** for state management (user store + chat store)
- **CSS Modules** (`.module.css`) for scoped component styles
- **No React Router** — single-view chat app, no client-side routing needed
- **No TypeScript** — plain JavaScript to keep the learning curve low
- **No UI library / Tailwind** — uses existing CSS custom properties
- **Native fetch** — no axios dependency
- **Two-phase chat flow** — `/api/discover` identifies the tool, `/api/chat` executes it
- **`/api/me` endpoint** replaces Jinja2 template variables for user context
- **Login page stays Jinja2** — server-side auth mode conditionals, no migration needed
- **`forwardRef` + `useImperativeHandle`** for ChatInput.fillText() (sidebar query fill)
- **`createPortal`** for PathFullscreen overlay

## Dependencies

```json
{
  "dependencies": { "react": "^18.3", "react-dom": "^18.3", "zustand": "^4.5" },
  "devDependencies": { "@vitejs/plugin-react": "^4.3", "vite": "^5.4" }
}
```
