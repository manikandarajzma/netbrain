#!/usr/bin/env python3
"""
generate_panorama_test_data.py

Creates large-scale test data in Panorama to simulate production load:
  - N address objects    (default 6 000)
  - N address groups     (default 6 000, each holding 1-5 address objects)
  - M security policies  (default 2 000, each referencing 1-3 address groups)

All objects are created in the device group you specify (or "shared" if blank).
The script commits once at the end — do NOT run against a live production
Panorama unless you are certain you want to add thousands of candidate objects.

Usage:
    python scripts/generate_panorama_test_data.py \
        --url https://panorama.example.com \
        --user admin --password secret \
        --device-group LoadTestDG \
        --addr-objects 6000 --addr-groups 6000 --policies 2000

Dry-run (print XML, no changes):
    python scripts/generate_panorama_test_data.py ... --dry-run

Requirements:
    pip install requests urllib3
"""

import argparse
import ipaddress
import random
import sys
import time
import urllib3
from typing import Optional

import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Panorama API helpers
# ---------------------------------------------------------------------------

def get_api_key(base_url: str, username: str, password: str, verify_ssl: bool) -> str:
    url = f"{base_url}/api/?type=keygen&user={username}&password={password}"
    resp = requests.get(url, verify=verify_ssl, timeout=15)
    resp.raise_for_status()
    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.text)
    key = root.findtext('.//key')
    if not key:
        raise RuntimeError(f"Keygen failed: {resp.text[:200]}")
    return key.strip()


def panorama_set(
    base_url: str,
    api_key: str,
    xpath: str,
    element: str,
    verify_ssl: bool,
    dry_run: bool = False,
) -> None:
    """Push a single set command to the Panorama candidate config."""
    if dry_run:
        print(f"DRY-RUN  xpath={xpath}\n         element={element}\n")
        return
    import urllib.parse
    url = (
        f"{base_url}/api/"
        f"?type=config&action=set"
        f"&xpath={urllib.parse.quote(xpath)}"
        f"&element={urllib.parse.quote(element)}"
        f"&key={api_key}"
    )
    resp = requests.post(url, verify=verify_ssl, timeout=30)
    resp.raise_for_status()
    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.text)
    status = root.get('status')
    if status != 'success':
        raise RuntimeError(f"API error: {resp.text[:400]}")


def panorama_commit(base_url: str, api_key: str, verify_ssl: bool, dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN  commit skipped")
        return
    url = f"{base_url}/api/?type=commit&cmd=<commit></commit>&key={api_key}"
    resp = requests.post(url, verify=verify_ssl, timeout=60)
    resp.raise_for_status()
    print(f"Commit response: {resp.text[:300]}")


# ---------------------------------------------------------------------------
# XPath builders
# ---------------------------------------------------------------------------

def _address_xpath(device_group: Optional[str], obj_name: str) -> str:
    if device_group:
        return (
            f"/config/devices/entry[@name='localhost.localdomain']"
            f"/device-group/entry[@name='{device_group}']"
            f"/address/entry[@name='{obj_name}']"
        )
    return f"/config/shared/address/entry[@name='{obj_name}']"


def _address_group_xpath(device_group: Optional[str], group_name: str) -> str:
    if device_group:
        return (
            f"/config/devices/entry[@name='localhost.localdomain']"
            f"/device-group/entry[@name='{device_group}']"
            f"/address-group/entry[@name='{group_name}']"
        )
    return f"/config/shared/address-group/entry[@name='{group_name}']"


def _security_rule_xpath(device_group: Optional[str], rule_name: str) -> str:
    if device_group:
        return (
            f"/config/devices/entry[@name='localhost.localdomain']"
            f"/device-group/entry[@name='{device_group}']"
            f"/pre-rulebase/security/rules/entry[@name='{rule_name}']"
        )
    return f"/config/shared/pre-rulebase/security/rules/entry[@name='{rule_name}']"


# ---------------------------------------------------------------------------
# Element builders
# ---------------------------------------------------------------------------

def _addr_object_element(ip: str) -> str:
    """<ip-netmask> address object XML element."""
    return f"<ip-netmask>{ip}</ip-netmask>"


def _addr_group_element(members: list[str]) -> str:
    """Static address group XML element."""
    member_xml = "".join(f"<member>{m}</member>" for m in members)
    return f"<static>{member_xml}</static>"


def _security_rule_element(src_groups: list[str], dst_groups: list[str]) -> str:
    """Minimal allow security rule referencing address groups."""
    src_xml = "".join(f"<member>{g}</member>" for g in src_groups)
    dst_xml = "".join(f"<member>{g}</member>" for g in dst_groups)
    return (
        f"<from><member>any</member></from>"
        f"<to><member>any</member></to>"
        f"<source>{src_xml}</source>"
        f"<destination>{dst_xml}</destination>"
        f"<application><member>any</member></application>"
        f"<service><member>application-default</member></service>"
        f"<action>allow</action>"
    )


# ---------------------------------------------------------------------------
# IP address generator
# ---------------------------------------------------------------------------

def generate_ips(count: int) -> list[str]:
    """Generate `count` unique /32 IPs in the 10.0.0.0/8 range."""
    base = ipaddress.IPv4Network("10.0.0.0/8")
    hosts = list(base.hosts())
    if count > len(hosts):
        raise ValueError(f"Requested {count} IPs but only {len(hosts)} available in 10.0.0.0/8")
    return [str(h) for h in random.sample(hosts, count)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Panorama load-test data")
    parser.add_argument("--url", required=True, help="Panorama base URL (e.g. https://10.1.1.1)")
    parser.add_argument("--user", required=True, help="Panorama admin username")
    parser.add_argument("--password", required=True, help="Panorama admin password")
    parser.add_argument("--device-group", default="", help="Device group name (blank = shared)")
    parser.add_argument("--addr-objects", type=int, default=6000, help="Number of address objects to create")
    parser.add_argument("--addr-groups", type=int, default=6000, help="Number of address groups to create")
    parser.add_argument("--policies", type=int, default=2000, help="Number of security policies to create")
    parser.add_argument("--batch-size", type=int, default=100, help="Commit every N objects (0 = commit once at end)")
    parser.add_argument("--prefix", default="lt", help="Name prefix for all generated objects (default: 'lt')")
    parser.add_argument("--no-ssl-verify", action="store_true", help="Skip SSL certificate verification")
    parser.add_argument("--dry-run", action="store_true", help="Print API calls without executing them")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    device_group = args.device_group.strip() or None
    verify_ssl = not args.no_ssl_verify
    prefix = args.prefix
    n_objects = args.addr_objects
    n_groups = args.addr_groups
    n_policies = args.policies
    batch_size = args.batch_size

    # ------------------------------------------------------------------
    # Authenticate
    # ------------------------------------------------------------------
    if args.dry_run:
        api_key = "DRY-RUN-KEY"
        print(f"[dry-run] Skipping authentication")
    else:
        print(f"Authenticating to {base_url} as {args.user} ...")
        api_key = get_api_key(base_url, args.user, args.password, verify_ssl)
        print(f"Got API key (first 8 chars): {api_key[:8]}...")

    dg_label = device_group or "shared"
    print(f"\nTarget: {dg_label}")
    print(f"  Address objects : {n_objects}")
    print(f"  Address groups  : {n_groups}")
    print(f"  Security policies: {n_policies}")
    print(f"  Batch size       : {batch_size if batch_size > 0 else 'end-only commit'}\n")

    # ------------------------------------------------------------------
    # Step 1: Address objects
    # ------------------------------------------------------------------
    print(f"[1/3] Generating {n_objects} address objects ...")
    ips = generate_ips(n_objects)
    obj_names: list[str] = []
    created = 0

    for i, ip in enumerate(ips):
        obj_name = f"{prefix}-obj-{i:05d}"
        obj_names.append(obj_name)
        xpath = _address_xpath(device_group, obj_name)
        element = _addr_object_element(ip)
        try:
            panorama_set(base_url, api_key, xpath, element, verify_ssl, args.dry_run)
            created += 1
        except Exception as exc:
            print(f"  WARNING: Failed to create {obj_name}: {exc}", file=sys.stderr)

        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{n_objects} address objects submitted")

        if batch_size > 0 and (i + 1) % batch_size == 0:
            panorama_commit(base_url, api_key, verify_ssl, args.dry_run)

    print(f"  Created {created}/{n_objects} address objects.\n")

    # ------------------------------------------------------------------
    # Step 2: Address groups
    # ------------------------------------------------------------------
    print(f"[2/3] Generating {n_groups} address groups (1-5 members each) ...")
    group_names: list[str] = []
    created = 0

    for i in range(n_groups):
        grp_name = f"{prefix}-grp-{i:05d}"
        group_names.append(grp_name)

        # Pick 1-5 random address objects as members
        member_count = random.randint(1, min(5, len(obj_names)))
        members = random.sample(obj_names, member_count)

        xpath = _address_group_xpath(device_group, grp_name)
        element = _addr_group_element(members)
        try:
            panorama_set(base_url, api_key, xpath, element, verify_ssl, args.dry_run)
            created += 1
        except Exception as exc:
            print(f"  WARNING: Failed to create {grp_name}: {exc}", file=sys.stderr)

        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{n_groups} address groups submitted")

        if batch_size > 0 and (i + 1) % batch_size == 0:
            panorama_commit(base_url, api_key, verify_ssl, args.dry_run)

    print(f"  Created {created}/{n_groups} address groups.\n")

    # ------------------------------------------------------------------
    # Step 3: Security policies
    # ------------------------------------------------------------------
    print(f"[3/3] Generating {n_policies} security policies ...")
    created = 0

    for i in range(n_policies):
        rule_name = f"{prefix}-rule-{i:05d}"

        # Each policy references 1-2 source groups and 1-2 destination groups
        src_count = random.randint(1, min(2, len(group_names)))
        dst_count = random.randint(1, min(2, len(group_names)))
        src_groups = random.sample(group_names, src_count)
        dst_groups = random.sample(group_names, dst_count)

        xpath = _security_rule_xpath(device_group, rule_name)
        element = _security_rule_element(src_groups, dst_groups)
        try:
            panorama_set(base_url, api_key, xpath, element, verify_ssl, args.dry_run)
            created += 1
        except Exception as exc:
            print(f"  WARNING: Failed to create {rule_name}: {exc}", file=sys.stderr)

        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{n_policies} policies submitted")

        if batch_size > 0 and (i + 1) % batch_size == 0:
            panorama_commit(base_url, api_key, verify_ssl, args.dry_run)

    print(f"  Created {created}/{n_policies} policies.\n")

    # ------------------------------------------------------------------
    # Final commit
    # ------------------------------------------------------------------
    print("Final commit ...")
    panorama_commit(base_url, api_key, verify_ssl, args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
