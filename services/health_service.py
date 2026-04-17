"""Health snapshot owner for Atlas backend and dependent services."""
from __future__ import annotations

import os
from typing import Any

import aiohttp

try:
    from atlas.security.auth import AUTH_MODE
    from atlas.agents.agent_factory import agent_factory
    from atlas.tools.shared import OLLAMA_BASE_URL
except ImportError:
    from security.auth import AUTH_MODE  # type: ignore
    from agents.agent_factory import agent_factory  # type: ignore
    from tools.shared import OLLAMA_BASE_URL  # type: ignore


class HealthService:
    """Owns live health snapshots for Atlas and key backend dependencies."""

    async def _check_mcp(self) -> dict[str, Any]:
        host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
        port = os.getenv("MCP_SERVER_PORT", "8765")
        mcp_url = f"http://{host}:{port}/health"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(mcp_url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status != 200:
                        return {"status": f"error ({resp.status})", "url": mcp_url, "tools_registered": None}
                    data = await resp.json()
                    return {
                        "status": data.get("status", "ok"),
                        "url": mcp_url,
                        "tools_registered": data.get("tools_registered"),
                    }
        except Exception:
            return {"status": "unreachable", "url": mcp_url, "tools_registered": None}

    async def _check_ollama(self) -> dict[str, Any]:
        base = OLLAMA_BASE_URL.rstrip("/")
        models_url = f"{base}/models"
        configured_models = agent_factory.configured_models()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status != 200:
                        return {
                            "status": f"error ({resp.status})",
                            "url": OLLAMA_BASE_URL,
                            "models": configured_models,
                            "missing_models": list(configured_models.values()),
                            "all_models_available": None,
                        }
                    data = await resp.json()
                    available = [m.get("id", "") for m in data.get("data", [])]
                    missing_models = sorted(
                        {
                            model
                            for model in configured_models.values()
                            if not any(
                                candidate == model or candidate.split(":")[0] == model.split(":")[0]
                                for candidate in available
                            )
                        }
                    )
                    all_models_available = not missing_models
                    return {
                        "status": "ok" if all_models_available else "model_not_found",
                        "url": OLLAMA_BASE_URL,
                        "models": configured_models,
                        "missing_models": missing_models,
                        "all_models_available": all_models_available,
                    }
        except Exception:
            return {
                "status": "unreachable",
                "url": OLLAMA_BASE_URL,
                "models": configured_models,
                "missing_models": list(configured_models.values()),
                "all_models_available": None,
            }

    async def _check_nornir(self) -> dict[str, Any]:
        url = "http://127.0.0.1:8006/devices"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status != 200:
                        return {"status": f"error ({resp.status})", "url": url, "device_count": None}
                    data = await resp.json()
                    device_count = len(data) if isinstance(data, list) else None
                    return {"status": "ok", "url": url, "device_count": device_count}
        except Exception:
            return {"status": "unreachable", "url": url, "device_count": None}

    def _overall_status(self, mcp: dict[str, Any], ollama: dict[str, Any], nornir: dict[str, Any]) -> dict[str, str]:
        statuses = [mcp.get("status"), ollama.get("status"), nornir.get("status")]
        if all(status == "ok" for status in statuses):
            return {"status": "healthy", "label": "All systems OK"}
        if any(status == "unreachable" for status in statuses):
            return {"status": "degraded", "label": "Dependency offline"}
        if any(isinstance(status, str) and status.startswith("error") for status in statuses):
            return {"status": "degraded", "label": "Dependency error"}
        if ollama.get("status") == "model_not_found":
            return {"status": "degraded", "label": "Model not found"}
        return {"status": "degraded", "label": "System issue"}

    async def build_snapshot(self) -> dict[str, Any]:
        mcp = await self._check_mcp()
        ollama = await self._check_ollama()
        nornir = await self._check_nornir()
        overall = self._overall_status(mcp, ollama, nornir)
        return {
            "status": "ok",
            "auth_mode": AUTH_MODE,
            "overall": overall,
            "services": {
                "mcp": mcp,
                "ollama": ollama,
                "nornir": nornir,
            },
        }


health_service = HealthService()
