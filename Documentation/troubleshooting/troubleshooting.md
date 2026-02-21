# Troubleshooting

Common issues and how to fix them.

---

## UI not up to date (missing buttons, old layout, changes not visible)

The web app can be served in two ways:

1. **React build** — FastAPI serves the built files from `frontend/dist/`. If that folder exists, you see the React UI.
2. **Static fallback** — If `frontend/dist/` does not exist, FastAPI serves the Jinja2/static UI.

After you (or someone else) change the frontend code, the **built** files in `dist/` do not update automatically. You must rebuild so the UI at **http://localhost:8000** shows the latest changes.

### Rebuild steps

From the project root:

```bash
cd frontend
npm install
npm run build
```

- **`npm install`** — Installs dependencies (e.g. Vite, React). Needed after cloning or when `node_modules` is missing.
- **`npm run build`** — Builds the React app into `frontend/dist/`. Run this whenever you change frontend code and want the production UI to reflect it.

Then restart the FastAPI server if it was already running, or refresh the browser (and do a hard refresh if needed: **Cmd+Shift+R** on Mac, **Ctrl+Shift+R** on Windows/Linux).

### Development without rebuilding every time

For active UI work, use the Vite dev server so changes apply immediately:

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** (log in at http://localhost:8000/login first so the session cookie is set). Edits to `.jsx` and `.module.css` will hot-reload; no `npm run build` needed until you want to update what's served at port 8000.

---

## Other issues

For chat/API issues, MCP server, Ollama, NetBox, or auth, see the main [README Troubleshooting](../../README.md#troubleshooting) section.
