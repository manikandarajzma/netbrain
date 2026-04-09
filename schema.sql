-- Atlas network device data schema
-- Run: psql atlas < schema.sql

-- Enable btree_gist for INET/CIDR indexing
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- ---------------------------------------------------------------------------
-- Routing tables
-- Stores parsed "show ip route vrf all" output per device.
-- Longest-prefix match query: WHERE '11.0.0.1'::inet << prefix ORDER BY masklen(prefix) DESC
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS routing_table (
    device          TEXT        NOT NULL,
    vrf             TEXT        NOT NULL DEFAULT 'default',
    prefix          CIDR        NOT NULL,
    next_hop        INET,                       -- NULL for connected/local routes
    egress_interface TEXT,
    protocol        TEXT,                       -- ospf, bgp, eigrp, static, connected
    admin_distance  SMALLINT,
    metric          INTEGER,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (device, vrf, prefix)
);

CREATE INDEX IF NOT EXISTS idx_routing_prefix
    ON routing_table USING GIST (prefix inet_ops);

CREATE INDEX IF NOT EXISTS idx_routing_device_vrf
    ON routing_table (device, vrf);

-- ---------------------------------------------------------------------------
-- ARP tables
-- Stores parsed "show ip arp vrf all" output per device.
-- Used to: (1) identify VRF for a source IP, (2) resolve next-hop IP → device
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS arp_table (
    device          TEXT        NOT NULL,
    vrf             TEXT        NOT NULL DEFAULT 'default',
    ip              INET        NOT NULL,
    mac             TEXT,
    interface       TEXT,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (device, vrf, ip)
);

CREATE INDEX IF NOT EXISTS idx_arp_ip
    ON arp_table (ip);

-- ---------------------------------------------------------------------------
-- Device inventory
-- Management IPs and platform info — populated from NetBox.
-- Used by Nornir to know where to SSH.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS devices (
    hostname        TEXT        PRIMARY KEY,
    mgmt_ip         INET        NOT NULL,
    platform        TEXT        NOT NULL,   -- cisco_ios, cisco_nxos, arista_eos, cisco_iosxr
    site            TEXT,
    role            TEXT,                   -- router, switch, firewall
    last_seen       TIMESTAMPTZ,
    synced_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_devices_mgmt_ip
    ON devices (mgmt_ip);

-- ---------------------------------------------------------------------------
-- Historical snapshots
-- Accumulate every collection run. Never deleted — used for "what was X yesterday?" queries.
-- Partitioning by month recommended if data grows large.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS arp_history (
    device          TEXT        NOT NULL,
    vrf             TEXT        NOT NULL DEFAULT 'default',
    ip              INET        NOT NULL,
    mac             TEXT,
    interface       TEXT,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_arp_history_device_time
    ON arp_history (device, collected_at DESC);

CREATE INDEX IF NOT EXISTS idx_arp_history_ip_time
    ON arp_history (ip, collected_at DESC);

CREATE TABLE IF NOT EXISTS routing_history (
    device          TEXT        NOT NULL,
    vrf             TEXT        NOT NULL DEFAULT 'default',
    prefix          CIDR        NOT NULL,
    next_hop        INET,
    egress_interface TEXT,
    protocol        TEXT,
    admin_distance  SMALLINT,
    metric          INTEGER,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_routing_history_device_time
    ON routing_history (device, collected_at DESC);

CREATE TABLE IF NOT EXISTS mac_history (
    device          TEXT        NOT NULL,
    mac             TEXT        NOT NULL,
    vlan            SMALLINT,
    interface       TEXT,
    entry_type      TEXT,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mac_history_device_time
    ON mac_history (device, collected_at DESC);

-- ---------------------------------------------------------------------------
-- Interface IPs
-- Maps device+interface → IP address.
-- Used to resolve a next-hop IP to the device that owns it.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS interface_ips (
    device          TEXT        NOT NULL,
    interface       TEXT        NOT NULL,
    ip              INET        NOT NULL,
    prefix_len      SMALLINT    NOT NULL,
    vrf             TEXT        NOT NULL DEFAULT 'default',
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (device, interface, ip)
);

CREATE INDEX IF NOT EXISTS idx_interface_ips_ip
    ON interface_ips (ip);

-- ---------------------------------------------------------------------------
-- MAC address table
-- Stores parsed "show mac address-table" per device.
-- Used for last-hop L2 resolution (end host → switch port).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mac_table (
    device          TEXT        NOT NULL,
    mac             TEXT        NOT NULL,
    vlan            SMALLINT,
    interface       TEXT,
    entry_type      TEXT,                       -- dynamic, static
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (device, mac, vlan)
);

CREATE INDEX IF NOT EXISTS idx_mac_table_mac
    ON mac_table (mac);

-- ---------------------------------------------------------------------------
-- OSPF neighbor table
-- Current OSPF adjacency state per device, refreshed each collection run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ospf_neighbors (
    device          TEXT        NOT NULL,
    vrf             TEXT        NOT NULL DEFAULT 'default',
    instance_id     TEXT        NOT NULL DEFAULT '1',
    router_id       TEXT        NOT NULL,
    neighbor_ip     INET,
    interface       TEXT,
    state           TEXT,                       -- full, 2way, init, down, etc.
    area            TEXT,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (device, vrf, instance_id, router_id)
);

CREATE INDEX IF NOT EXISTS idx_ospf_neighbors_device
    ON ospf_neighbors (device, collected_at DESC);

-- ---------------------------------------------------------------------------
-- OSPF neighbor history
-- Accumulates every collection run — never deleted.
-- Used to detect when adjacencies appeared/disappeared.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ospf_history (
    device          TEXT        NOT NULL,
    vrf             TEXT        NOT NULL DEFAULT 'default',
    instance_id     TEXT        NOT NULL DEFAULT '1',
    router_id       TEXT        NOT NULL,
    neighbor_ip     INET,
    interface       TEXT,
    state           TEXT,
    area            TEXT,
    collected_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ospf_history_device_time
    ON ospf_history (device, collected_at DESC);

-- ---------------------------------------------------------------------------
-- Collection runs
-- Tracks when each device was last collected and whether it succeeded.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS collection_runs (
    id              BIGSERIAL   PRIMARY KEY,
    device          TEXT        NOT NULL,
    run_type        TEXT        NOT NULL,   -- 'routes', 'arp', 'logs'
    status          TEXT        NOT NULL,   -- 'success', 'failed', 'timeout'
    duration_ms     INTEGER,
    error           TEXT,
    ran_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_collection_device_type
    ON collection_runs (device, run_type, ran_at DESC);
