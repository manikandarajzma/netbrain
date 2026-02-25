# Atlas Troubleshooting

## Service won't start

```bash
# See why a service failed to start
sudo systemctl status atlas-web
journalctl -u atlas-web -n 50 --no-pager
```

## Find all errors across both services since last boot

```bash
journalctl -u atlas-web -u atlas-mcp -p err -b --no-pager
```

## Search application logs for a specific IP or error message

```bash
grep "10.0.0.1" /var/log/atlas/atlas_web.log
grep "ERROR" /var/log/atlas/mcp_server.log | tail -20
```

## Check how many times a service has restarted

```bash
sudo systemctl status atlas-web | grep -E "Active|restarts"
```

## Watch both logs at the same time

```bash
tail -f /var/log/atlas/atlas_web.log /var/log/atlas/mcp_server.log
```

## MCP server not reachable from web app

```bash
# Confirm MCP server is listening on the expected port
ss -tlnp | grep 8765
# Check MCP server health directly
curl http://127.0.0.1:8765/health
```

## Session or login issues

```bash
# Filter auth-related log lines
grep -E "auth|session|group|login|OIDC" /var/log/atlas/atlas_web.log | tail -30
```
