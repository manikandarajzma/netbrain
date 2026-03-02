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

---

## Case Studies

### Chat history encryption error (`TypeError: Cannot convert str instance to a buffer`)

**Symptom**

Sending a chat message returned a `500 Internal Server Error`. The UI showed no useful message.

**How we identified it — Browser DevTools**

1. Open **DevTools** → **Network** tab.
2. Send a message to trigger the error.
3. Click the failed request (e.g. `POST /api/chat`).
4. Select the **Response** sub-tab.

The raw response body contained the full Python traceback:

```
File "/opt/atlas/chat_history.py", line 120, in _write_json_file
    ct = AESGCM(_get_aes_key()).encrypt(nonce, plaintext, None)
TypeError: Cannot convert "<class 'str'>" instance to a buffer.
Did you mean to pass a bytestring instead?
```

Without the Response tab the error would have been invisible — the browser only shows a blank error page.

**Root cause**

`_get_aes_key()` was refactored to retrieve the key via `get_creds('CHAT-ENCRYPTION-KEY')`. Azure Key Vault's SDK (`SecretClient.get_secret().value`) always returns a Python `str`. The `AESGCM` constructor requires `bytes`.

The refactor dropped the decoding step that converted the stored hex/base64 string into raw bytes.

**Fix**

After calling `get_creds`, decode the string to 32 raw bytes before passing it to `AESGCM`:

```python
value = get_creds('CHAT-ENCRYPTION-KEY').strip()
if len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value):
    key = bytes.fromhex(value)   # stored as hex
else:
    key = base64.b64decode(value) # stored as base64
```
