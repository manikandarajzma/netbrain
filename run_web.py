"""
Launcher for the Atlas web app. Ensures the parent directory is on sys.path
so that the atlas package is found when running from the project root.
Configures logging to go to log files only (no console output).
"""
import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_parent = _root.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

# Log to file only (no console).
_log_file = _root / "atlas_web.log"
_file_handler = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_root_logger = logging.getLogger()
for _h in _root_logger.handlers[:]:
    _root_logger.removeHandler(_h)
_root_logger.addHandler(_file_handler)
_root_logger.setLevel(logging.INFO)
for _name, _logr in logging.Logger.manager.loggerDict.items():
    if isinstance(_logr, logging.Logger):
        for _h in _logr.handlers[:]:
            if isinstance(_h, logging.StreamHandler):
                _logr.removeHandler(_h)

from atlas.app import app  # noqa: E402

__all__ = ["app"]

# Run with reload and log config built in (no CLI flags needed)
if __name__ == "__main__":
    import json
    import uvicorn

    _host, _port = "0.0.0.0", 8000
    _log_config_path = _root / "log_config.json"
    if _log_config_path.exists():
        _log_cfg = json.loads(_log_config_path.read_text())
        _log_cfg["handlers"]["file"]["filename"] = str(_root / "atlas_web.log")
    else:
        _log_cfg = None

    print(
        f"Atlas web server starting at http://{_host}:{_port} (reload on). Logs: atlas_web.log\n"
        "Server runs in foreground; press Ctrl+C to stop. If the prompt returns, check atlas_web.log for errors.",
        flush=True,
    )
    try:
        uvicorn.run(
            "run_web:app",
            host=_host,
            port=_port,
            reload=True,
            log_config=_log_cfg,
        )
    except Exception as e:
        print(f"Server failed to start: {e}", file=sys.stderr, flush=True)
        raise
