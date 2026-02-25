# Atlas Operations Guide

## Running in Production (systemd)

Atlas has two processes that must both be running:

- **MCP server** (`mcp_server.py`) — serves tools over HTTP to the web app
- **Web app** (`run_web:app`) — FastAPI/uvicorn, serves the browser UI

Both are managed as systemd services so they start on boot and restart automatically on failure.

---

### Service Files

Create the following two files as root:

**`/etc/systemd/system/atlas-mcp.service`**

```ini
[Unit]
Description=Atlas MCP Server
After=network.target

[Service]
User=atlas
WorkingDirectory=/opt/atlas
ExecStart=/opt/atlas/.venv/bin/python mcp_server.py
Restart=always
RestartSec=5
EnvironmentFile=/opt/atlas/.env
LogsDirectory=atlas

[Install]
WantedBy=multi-user.target
```

| Line | Meaning |
|------|---------|
| `Description=` | Human-readable label shown in `systemctl status` output |
| `After=network.target` | Don't start until the network is up |
| `User=atlas` | Run the process as this OS user — never run as root |
| `WorkingDirectory=` | Sets the working directory before launching the process; relative file paths in the app resolve from here |
| `ExecStart=` | The exact command systemd runs — uses the virtualenv Python so all installed packages are available |
| `Restart=always` | Automatically restart if the process exits for any reason (crash, OOM, etc.) |
| `RestartSec=5` | Wait 5 seconds before restarting — prevents a crash loop from hammering the system |
| `EnvironmentFile=` | Loads every `KEY=VALUE` line from `/opt/atlas/.env` into the process environment before startup; this is how all secrets and config reach the app |
| `LogsDirectory=atlas` | Tells systemd to create `/var/log/atlas/` (owned by `User`) if it doesn't exist; the app writes `mcp_server.log` there |
| `WantedBy=multi-user.target` | Enables the service to start at normal (non-graphical) boot |

---

**`/etc/systemd/system/atlas-web.service`**

```ini
[Unit]
Description=Atlas Web App
After=network.target atlas-mcp.service
Wants=atlas-mcp.service

[Service]
User=atlas
WorkingDirectory=/opt/atlas
ExecStart=/opt/atlas/.venv/bin/uvicorn run_web:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
EnvironmentFile=/opt/atlas/.env
LogsDirectory=atlas

[Install]
WantedBy=multi-user.target
```

| Line | Meaning |
|------|---------|
| `After=network.target atlas-mcp.service` | Start only after both the network and the MCP server are up |
| `Wants=atlas-mcp.service` | Declares a soft dependency — if `atlas-mcp` is not running, systemd will try to start it first |
| `ExecStart=` | Runs uvicorn with the `run_web:app` ASGI application on all interfaces (`0.0.0.0`) port 8000 with 2 worker processes |
| `--workers 2` | Two worker processes handle requests in parallel; increase for higher concurrency |
| All other lines | Same meaning as in `atlas-mcp.service` above |

> Adjust `User`, `WorkingDirectory`, and the `.venv` path to match your deployment layout.

---

### Enable and Start

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now atlas-mcp atlas-web
```

---

### Common Commands

| Action | Command |
|--------|---------|
| Check status | `sudo systemctl status atlas-mcp atlas-web` |
| Restart both | `sudo systemctl restart atlas-mcp atlas-web` |
| Stop both | `sudo systemctl stop atlas-web atlas-mcp` |
| View live logs (web) | `journalctl -u atlas-web -f` |
| View live logs (mcp) | `journalctl -u atlas-mcp -f` |
| View recent errors | `journalctl -u atlas-web -p err -n 50` |

---

### Notes

- Stop `atlas-web` before `atlas-mcp` — the web app depends on the MCP server.
- The `.env` file is loaded by systemd via `EnvironmentFile`. Ensure it is readable only by the `atlas` user (`chmod 600 /opt/atlas/.env`).
- Log files are written to `/var/log/atlas/` (`mcp_server.log`, `atlas_web.log`). The `LogsDirectory=atlas` directive in the service files instructs systemd to create this directory with correct ownership automatically.
- Under systemd, stdout/stderr is also captured by journald and available via `journalctl`. The log files contain application-level logging only.

---

For troubleshooting commands see [troubleshooting/troubleshooting.md](../troubleshooting/troubleshooting.md).
