"""
Nornir device data collector — runs offline, periodically via cron or manually.

Connects to network devices via SSH, collects routing + ARP tables,
and upserts them into PostgreSQL for path tracing queries.

Usage:
    uv run python scripts/collect_devices.py

Arista EOS devices use JSON-structured output (show ip route vrf all | json)
for reliable parsing without TextFSM.
"""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from repo root so DATABASE_URL is available when run directly
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("atlas.collector")

# ---------------------------------------------------------------------------
# Static device inventory — replace with NetBox pull when ready
# ---------------------------------------------------------------------------
DEVICES = [
    {"hostname": "arista1", "host": "127.0.0.1", "port": 6022, "platform": "arista_eos"},
    {"hostname": "arista2", "host": "127.0.0.1", "port": 6023, "platform": "arista_eos"},
    {"hostname": "arista3", "host": "127.0.0.1", "port": 6024, "platform": "arista_eos"},
]

# SSH credentials — override via env vars
import os
SSH_USER     = os.getenv("DEVICE_SSH_USER", "admin")
SSH_PASSWORD = os.getenv("DEVICE_SSH_PASSWORD", "admin")


# ---------------------------------------------------------------------------
# Nornir inventory built dynamically (no files needed)
# ---------------------------------------------------------------------------
def _build_nornir():
    from nornir.core import Nornir
    from nornir.core.inventory import (
        Inventory, Hosts, Host, Groups, Defaults, ConnectionOptions,
    )
    from nornir.core.plugins.runners import RunnersPluginRegister
    from nornir.core.plugins.connections import ConnectionPluginRegister
    RunnersPluginRegister.auto_register()
    ConnectionPluginRegister.auto_register()
    runner_cls = RunnersPluginRegister.available["threaded"]

    hosts = {}
    for d in DEVICES:
        hosts[d["hostname"]] = Host(
            name=d["hostname"],
            hostname=d["host"],
            port=d["port"],
            username=SSH_USER,
            password=SSH_PASSWORD,
            platform=d["platform"],
            connection_options={
                "netmiko": ConnectionOptions(
                    extras={
                        "timeout": 30,
                        "global_delay_factor": 1,
                    }
                )
            },
        )

    inventory = Inventory(hosts=Hosts(hosts), groups=Groups(), defaults=Defaults())
    runner = runner_cls(num_workers=10)
    return Nornir(inventory=inventory, runner=runner)


# ---------------------------------------------------------------------------
# Arista EOS parsers (JSON output)
# ---------------------------------------------------------------------------
def _parse_arista_interfaces(hostname: str, raw_json: str) -> list[dict]:
    """Parse 'show interfaces | json' into interface IP rows."""
    rows = []
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.warning("%s: failed to parse interfaces JSON: %s", hostname, e)
        return rows

    for intf_name, intf_data in data.get("interfaces", {}).items():
        vrf = intf_data.get("vrf", "default") or "default"
        for addr_entry in intf_data.get("interfaceAddress", []):
            primary = addr_entry.get("primaryIp", {})
            ip = primary.get("address")
            mask = primary.get("maskLen")
            if ip and ip != "0.0.0.0" and mask is not None:
                rows.append({
                    "device":    hostname,
                    "interface": intf_name,
                    "ip":        ip,
                    "prefix_len": mask,
                    "vrf":       vrf,
                })
    return rows


def _parse_arista_ospf(hostname: str, raw_json: str) -> list[dict]:
    """Parse 'show ip ospf neighbor | json' into OSPF neighbor rows."""
    rows = []
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.warning("%s: failed to parse OSPF neighbor JSON: %s", hostname, e)
        return rows

    for vrf_name, vrf_data in data.get("vrfs", {}).items():
        for inst_id, inst_data in vrf_data.get("instList", {}).items():
            for nbr in inst_data.get("ospfNeighborEntries", []):
                router_id = nbr.get("routerId")
                if not router_id:
                    continue
                rows.append({
                    "device":      hostname,
                    "vrf":         vrf_name,
                    "instance_id": inst_id,
                    "router_id":   router_id,
                    "neighbor_ip": nbr.get("interfaceAddress"),
                    "interface":   nbr.get("interfaceName"),
                    "state":       nbr.get("adjacencyState"),
                    "area":        nbr.get("details", {}).get("areaId"),
                })
    return rows


def _parse_arista_mac(hostname: str, raw_json: str) -> list[dict]:
    """Parse 'show mac address-table | json' into MAC table rows."""
    rows = []
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.warning("%s: failed to parse MAC table JSON: %s", hostname, e)
        return rows

    for entry in data.get("unicastTable", {}).get("tableEntries", []):
        mac = entry.get("macAddress")
        if not mac:
            continue
        rows.append({
            "device":     hostname,
            "mac":        mac,
            "vlan":       entry.get("vlanId"),
            "interface":  entry.get("interface"),
            "entry_type": entry.get("entryType", "dynamic"),
        })
    return rows


def _parse_arista_routes(hostname: str, raw_json: str) -> list[dict]:
    """Parse 'show ip route vrf all | json' into route rows."""
    rows = []
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.warning("%s: failed to parse route JSON: %s", hostname, e)
        return rows

    for vrf, vrf_data in data.get("vrfs", {}).items():
        for prefix, route_data in vrf_data.get("routes", {}).items():
            for via in route_data.get("vias", []):
                rows.append({
                    "device":          hostname,
                    "vrf":             vrf,
                    "prefix":          prefix,
                    "next_hop":        via.get("nexthopAddr") or None,
                    "egress_interface": via.get("interface") or None,
                    "protocol":        route_data.get("routeType", "").lower(),
                    "admin_distance":  route_data.get("preference"),
                    "metric":          route_data.get("metric"),
                })
    return rows


def _parse_arista_arp(hostname: str, raw_json: str) -> list[dict]:
    """Parse 'show ip arp vrf all | json' into ARP rows."""
    rows = []
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.warning("%s: failed to parse ARP JSON: %s", hostname, e)
        return rows

    for vrf, vrf_data in data.get("vrfs", {}).items():
        for entry in vrf_data.get("ipV4Neighbors", []):
            rows.append({
                "device":    hostname,
                "vrf":       vrf,
                "ip":        entry.get("address"),
                "mac":       entry.get("hwAddress"),
                "interface": entry.get("interface"),
            })
    return rows


# ---------------------------------------------------------------------------
# Per-device collection task (runs in Nornir thread pool)
# ---------------------------------------------------------------------------
def _collect_device(task):
    """Nornir task: collect routes + ARP from one device."""
    from nornir_netmiko.tasks import netmiko_send_command

    hostname = task.host.name
    results = {"hostname": hostname, "routes": [], "arp": [], "interfaces": [], "mac": [], "errors": []}

    commands = [
        ("routes",     "show ip route vrf all | json",    _parse_arista_routes),
        ("arp",        "show ip arp vrf all | json",       _parse_arista_arp),
        ("interfaces", "show interfaces | json",           _parse_arista_interfaces),
        ("mac",        "show mac address-table | json",    _parse_arista_mac),
        ("ospf",       "show ip ospf neighbor | json",     _parse_arista_ospf),
    ]

    for key, cmd, parser in commands:
        try:
            r = task.run(task=netmiko_send_command, command_string=cmd, name=key)
            results[key] = parser(hostname, r.result)
            logger.info("%s: got %d %s entries", hostname, len(results[key]), key)
        except Exception as e:
            results["errors"].append(f"{key}: {e}")
            logger.warning("%s: %s collection failed: %s", hostname, key, e)

    return results


# ---------------------------------------------------------------------------
# Database upsert
# ---------------------------------------------------------------------------
async def _upsert_routes(device: str, rows: list[dict]) -> None:
    from db import execute, executemany

    # Clear stale routes for this device before inserting fresh data
    await execute("DELETE FROM routing_table WHERE device = $1", device)
    if not rows:
        return
    await executemany(
        """
        INSERT INTO routing_table
            (device, vrf, prefix, next_hop, egress_interface, protocol, admin_distance, metric)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (device, vrf, prefix) DO UPDATE SET
            next_hop         = EXCLUDED.next_hop,
            egress_interface = EXCLUDED.egress_interface,
            protocol         = EXCLUDED.protocol,
            admin_distance   = EXCLUDED.admin_distance,
            metric           = EXCLUDED.metric,
            collected_at     = now()
        """,
        [
            (r["device"], r["vrf"], r["prefix"], r["next_hop"],
             r["egress_interface"], r["protocol"], r["admin_distance"], r["metric"])
            for r in rows
        ],
    )


async def _upsert_arp(device: str, rows: list[dict]) -> None:
    from db import execute, executemany

    await execute("DELETE FROM arp_table WHERE device = $1", device)
    if not rows:
        return
    await executemany(
        """
        INSERT INTO arp_table (device, vrf, ip, mac, interface)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (device, vrf, ip) DO UPDATE SET
            mac          = EXCLUDED.mac,
            interface    = EXCLUDED.interface,
            collected_at = now()
        """,
        [(r["device"], r["vrf"], r["ip"], r["mac"], r["interface"]) for r in rows],
    )


async def _upsert_device(device: dict) -> None:
    from db import execute
    await execute(
        """
        INSERT INTO devices (hostname, mgmt_ip, platform, last_seen)
        VALUES ($1, $2, $3, now())
        ON CONFLICT (hostname) DO UPDATE SET
            mgmt_ip   = EXCLUDED.mgmt_ip,
            platform  = EXCLUDED.platform,
            last_seen = now(),
            synced_at = now()
        """,
        device["hostname"], device["host"], device["platform"],
    )


async def _upsert_interfaces(device: str, rows: list[dict]) -> None:
    from db import execute, executemany
    await execute("DELETE FROM interface_ips WHERE device = $1", device)
    if not rows:
        return
    await executemany(
        """
        INSERT INTO interface_ips (device, interface, ip, prefix_len, vrf)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (device, interface, ip) DO UPDATE SET
            prefix_len   = EXCLUDED.prefix_len,
            vrf          = EXCLUDED.vrf,
            collected_at = now()
        """,
        [(r["device"], r["interface"], r["ip"], r["prefix_len"], r["vrf"]) for r in rows],
    )


async def _upsert_mac(device: str, rows: list[dict]) -> None:
    from db import execute, executemany
    await execute("DELETE FROM mac_table WHERE device = $1", device)
    if not rows:
        return
    await executemany(
        """
        INSERT INTO mac_table (device, mac, vlan, interface, entry_type)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (device, mac, vlan) DO UPDATE SET
            interface    = EXCLUDED.interface,
            entry_type   = EXCLUDED.entry_type,
            collected_at = now()
        """,
        [(r["device"], r["mac"], r["vlan"], r["interface"], r["entry_type"]) for r in rows],
    )


# ---------------------------------------------------------------------------
# History inserts — append every collection run, never delete
# ---------------------------------------------------------------------------
async def _insert_arp_history(rows: list[dict]) -> None:
    if not rows:
        return
    from db import executemany
    await executemany(
        """
        INSERT INTO arp_history (device, vrf, ip, mac, interface)
        VALUES ($1, $2, $3, $4, $5)
        """,
        [(r["device"], r["vrf"], r["ip"], r["mac"], r["interface"]) for r in rows],
    )


async def _insert_routing_history(rows: list[dict]) -> None:
    if not rows:
        return
    from db import executemany
    await executemany(
        """
        INSERT INTO routing_history
            (device, vrf, prefix, next_hop, egress_interface, protocol, admin_distance, metric)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        [
            (r["device"], r["vrf"], r["prefix"], r["next_hop"],
             r["egress_interface"], r["protocol"], r["admin_distance"], r["metric"])
            for r in rows
        ],
    )


async def _upsert_ospf(device: str, rows: list[dict]) -> None:
    from db import execute, executemany
    await execute("DELETE FROM ospf_neighbors WHERE device = $1", device)
    if not rows:
        return
    await executemany(
        """
        INSERT INTO ospf_neighbors
            (device, vrf, instance_id, router_id, neighbor_ip, interface, state, area)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (device, vrf, instance_id, router_id) DO UPDATE SET
            neighbor_ip  = EXCLUDED.neighbor_ip,
            interface    = EXCLUDED.interface,
            state        = EXCLUDED.state,
            area         = EXCLUDED.area,
            collected_at = now()
        """,
        [(r["device"], r["vrf"], r["instance_id"], r["router_id"],
          r["neighbor_ip"], r["interface"], r["state"], r["area"]) for r in rows],
    )


async def _insert_ospf_history(rows: list[dict]) -> None:
    if not rows:
        return
    from db import executemany
    await executemany(
        """
        INSERT INTO ospf_history
            (device, vrf, instance_id, router_id, neighbor_ip, interface, state, area)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        [(r["device"], r["vrf"], r["instance_id"], r["router_id"],
          r["neighbor_ip"], r["interface"], r["state"], r["area"]) for r in rows],
    )


async def _insert_mac_history(rows: list[dict]) -> None:
    if not rows:
        return
    from db import executemany
    await executemany(
        """
        INSERT INTO mac_history (device, mac, vlan, interface, entry_type)
        VALUES ($1, $2, $3, $4, $5)
        """,
        [(r["device"], r["mac"], r["vlan"], r["interface"], r["entry_type"]) for r in rows],
    )


async def _log_run(device: str, run_type: str, status: str, duration_ms: int, error: str = None):
    from db import execute
    await execute(
        "INSERT INTO collection_runs (device, run_type, status, duration_ms, error) VALUES ($1,$2,$3,$4,$5)",
        device, run_type, status, duration_ms, error,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_collection():
    logger.info("Starting device collection for %d devices...", len(DEVICES))
    t0 = time.perf_counter()

    nr = _build_nornir()
    result = nr.run(task=_collect_device)

    tasks = []
    for hostname, multi_result in result.items():
        device_result = multi_result[0].result
        if device_result is None:
            logger.error("%s: task returned no result", hostname)
            continue

        t_dev = int((time.perf_counter() - t0) * 1000)
        errors = device_result.get("errors", [])
        routes = device_result.get("routes", [])
        arp    = device_result.get("arp", [])

        interfaces = device_result.get("interfaces", [])
        mac        = device_result.get("mac", [])
        ospf       = device_result.get("ospf", [])

        dev_info = next(d for d in DEVICES if d["hostname"] == hostname)
        tasks.append(_upsert_device(dev_info))
        tasks.append(_upsert_routes(hostname, routes))
        tasks.append(_upsert_arp(hostname, arp))
        tasks.append(_upsert_interfaces(hostname, interfaces))
        tasks.append(_upsert_mac(hostname, mac))
        tasks.append(_upsert_ospf(hostname, ospf))
        # History snapshots — append without overwriting
        tasks.append(_insert_arp_history(arp))
        tasks.append(_insert_routing_history(routes))
        tasks.append(_insert_mac_history(mac))
        tasks.append(_insert_ospf_history(ospf))
        tasks.append(_log_run(
            hostname, "full",
            "success" if not errors else "partial",
            t_dev,
            "; ".join(errors) or None,
        ))

    await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t0
    logger.info("Collection complete in %.1fs", elapsed)


if __name__ == "__main__":
    asyncio.run(run_collection())
