"""
NetBox domain module – MCP tools for IPAM lookups used in path tracing.

Exposes the following MCP tools:
  - get_gateway_for_prefix  – returns the VIP (gateway) IP for a given prefix
  - get_prefix_for_ip       – returns the containing prefix for a given IP
  - get_ip_info             – returns NetBox metadata for a specific IP address

Authentication uses session-based login (NetBox 4.x API token auth is unavailable
in this environment). A module-level session is reused across calls.
"""

import ipaddress
import os
import threading
from typing import Optional

import requests

from tools.shared import mcp, setup_logging

logger = setup_logging(__name__)

NETBOX_URL      = os.getenv("NETBOX_URL", "http://localhost:8000")
NETBOX_USER     = os.getenv("NETBOX_USER", "admin")
NETBOX_PASSWORD = os.getenv("NETBOX_PASSWORD", "admin")

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

_session: Optional[requests.Session] = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    """Return an authenticated requests.Session, logging in if needed."""
    global _session
    with _session_lock:
        if _session is not None:
            return _session

        s = requests.Session()
        # Fetch CSRF token
        resp = s.get(f"{NETBOX_URL}/login/", timeout=10)
        resp.raise_for_status()
        csrf = s.cookies.get("csrftoken")
        if not csrf:
            raise RuntimeError("NetBox: could not retrieve CSRF token")

        # Login
        resp = s.post(
            f"{NETBOX_URL}/login/",
            data={
                "csrfmiddlewaretoken": csrf,
                "username": NETBOX_USER,
                "password": NETBOX_PASSWORD,
                "next": "/",
            },
            headers={"Referer": f"{NETBOX_URL}/login/"},
            allow_redirects=True,
            timeout=10,
        )
        # NetBox redirects to / on success; a 200 landing on /login/ means failure
        if "/login/" in resp.url:
            raise RuntimeError("NetBox: authentication failed — check NETBOX_USER/NETBOX_PASSWORD")

        logger.info("netbox: session established for %s", NETBOX_USER)
        _session = s
        return _session


def _api_get(path: str, **params) -> dict:
    """GET /api/<path> with automatic session refresh on 403."""
    global _session
    s = _get_session()
    url = f"{NETBOX_URL}/api/{path.lstrip('/')}"
    resp = s.get(url, params=params, timeout=10)
    if resp.status_code == 403:
        # Session expired — reset and retry once
        logger.warning("netbox: session expired, re-authenticating")
        with _session_lock:
            _session = None
        s = _get_session()
        resp = s.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_gateway_for_prefix(prefix: str) -> dict:
    """
    Return the gateway IP for a given IP prefix by finding the VIP-role address.

    Args:
        prefix: CIDR prefix to look up, e.g. "10.0.100.0/24" or derived
                automatically from a host IP like "10.0.100.55" (will expand
                to the /24 network).

    Returns:
        {"gateway": "10.0.100.1", "prefix": "10.0.100.0/24", "dns_name": "..."}
        or {"error": "..."} if not found.
    """
    try:
        net = ipaddress.ip_network(prefix, strict=False)
    except ValueError:
        return {"error": f"Invalid prefix or IP: {prefix!r}"}

    # If a host address was given (/32), resolve its containing prefix first
    if net.prefixlen == 32:
        prefix_info = get_prefix_for_ip.fn(str(net.network_address))
        if "error" in prefix_info:
            return {"error": f"Cannot determine prefix for {prefix}: {prefix_info['error']}"}
        net = ipaddress.ip_network(prefix_info["prefix"], strict=False)

    normalized = str(net)

    try:
        data = _api_get("ipam/ip-addresses/", parent=normalized, role="vip", limit=1)
    except Exception as e:
        return {"error": f"NetBox API error: {e}"}

    if not data.get("results"):
        return {"error": f"No VIP address found in prefix {normalized}"}

    ip_obj = data["results"][0]
    gw_ip = ip_obj["address"].split("/")[0]
    return {
        "gateway":  gw_ip,
        "prefix":   normalized,
        "dns_name": ip_obj.get("dns_name") or "",
        "netbox_id": ip_obj["id"],
    }


@mcp.tool()
def get_prefix_for_ip(ip: str) -> dict:
    """
    Return the most-specific NetBox prefix that contains the given IP.

    Args:
        ip: Host IP address, e.g. "10.0.100.55"

    Returns:
        {"prefix": "10.0.100.0/24", "vlan": "cust1_web", "role": "cust1_web"}
        or {"error": "..."} if not found.
    """
    try:
        ipaddress.ip_address(ip)
    except ValueError:
        return {"error": f"Invalid IP address: {ip!r}"}

    try:
        data = _api_get("ipam/prefixes/", contains=ip, limit=100)
    except Exception as e:
        return {"error": f"NetBox API error: {e}"}

    if not data.get("results"):
        return {"error": f"No prefix found containing {ip}"}

    # Pick most-specific (longest prefix length)
    best = max(data["results"], key=lambda p: ipaddress.ip_network(p["prefix"]).prefixlen)
    return {
        "prefix": best["prefix"],
        "vlan":   best["vlan"]["name"] if best.get("vlan") else None,
        "role":   best["role"]["name"] if best.get("role") else None,
        "vrf":    best["vrf"]["name"]  if best.get("vrf")  else None,
    }


@mcp.tool()
def get_ip_info(ip: str) -> dict:
    """
    Return NetBox metadata for a specific IP address.

    Args:
        ip: IP address with or without prefix length, e.g. "10.0.100.1" or "10.0.100.1/24"

    Returns:
        dict with address, role, status, dns_name, assigned device (if any).
    """
    # Normalise — strip prefix length for the lookup
    ip_only = ip.split("/")[0]
    try:
        ipaddress.ip_address(ip_only)
    except ValueError:
        return {"error": f"Invalid IP address: {ip!r}"}

    try:
        data = _api_get("ipam/ip-addresses/", address=ip_only, limit=5)
    except Exception as e:
        return {"error": f"NetBox API error: {e}"}

    if not data.get("results"):
        return {"error": f"IP {ip_only} not found in NetBox"}

    obj = data["results"][0]
    assigned = obj.get("assigned_object")
    return {
        "address":   obj["address"],
        "status":    obj["status"]["value"],
        "role":      obj["role"]["value"] if obj.get("role") else None,
        "dns_name":  obj.get("dns_name") or "",
        "assigned_to": assigned.get("display") if assigned else None,
        "netbox_url":  obj["display_url"],
    }
