const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const exampleQueriesEl = document.getElementById('example-queries');
let conversationHistory = [];

document.querySelectorAll('.example-chip').forEach(function(btn) {
    btn.addEventListener('click', function() {
        var q = this.getAttribute('data-query');
        if (q) { inputEl.value = q; inputEl.focus(); }
    });
});

function deviceTypeToIcon(deviceType) {
    if (!deviceType || typeof deviceType !== 'string') return null;
    var t = deviceType.toLowerCase().trim();
    if (t.indexOf('palo alto') !== -1 || (t.indexOf('firewall') !== -1 && t.indexOf('palo') !== -1)) return '/icons/paloalto_firewall.png';
    if (t.indexOf('arista') !== -1 || t.indexOf('switch') !== -1) return '/icons/arista_switch.png';
    if (t.indexOf('firewall') !== -1) return '/icons/paloalto_firewall.png';
    return null;
}
function deviceTypeToFallbackLabel(deviceType) {
    if (!deviceType || typeof deviceType !== 'string') return '?';
    var t = deviceType.toLowerCase().trim();
    if (t.indexOf('firewall') !== -1) return 'FW';
    if (t.indexOf('switch') !== -1 || t.indexOf('arista') !== -1) return 'SW';
    if (t.indexOf('hub') !== -1) return 'HUB';
    return '?';
}
function appendPathIconOrFallback(parent, deviceType) {
    var wrap = document.createElement('div');
    wrap.className = 'path-icon-wrap';
    var iconSrc = deviceTypeToIcon(deviceType);
    var fallbackLabel = deviceTypeToFallbackLabel(deviceType);
    if (iconSrc) {
        var img = document.createElement('img');
        img.className = 'path-icon';
        img.src = iconSrc;
        img.alt = deviceType || '';
        var fallback = document.createElement('span');
        fallback.className = 'path-icon-fallback';
        fallback.textContent = fallbackLabel;
        fallback.style.display = 'none';
        wrap.appendChild(img);
        wrap.appendChild(fallback);
        img.onerror = function() { img.style.display = 'none'; fallback.style.display = 'inline-flex'; };
    } else {
        var fb = document.createElement('span');
        fb.className = 'path-icon-fallback';
        fb.textContent = fallbackLabel;
        wrap.appendChild(fb);
    }
    parent.appendChild(wrap);
}

function normalizeInterface(s) {
    if (s == null || s === '') return '';
    s = String(s).trim();
    if (/^ethernet/i.test(s)) return 'Ethernet' + s.slice(8);
    if (/^eth/i.test(s)) return 'Ethernet' + s.slice(3);
    return s.charAt(0).toUpperCase() + s.slice(1);
}
function renderPathWithIcons(container, content) {
    var hops = content.path_hops;
    if (!hops || !hops.length) return;
    var wrap = document.createElement('div');
    wrap.className = 'path-visual';
    // Show status bar for failed/denied/unknown paths only (inline styles to bypass CSS cache)
    var _pathStatus = (content.path_status || '').toLowerCase();
    var _contentStatus = (content.status || '').toLowerCase();
    var _showStatus = (_pathStatus === 'failed' || _contentStatus === 'denied' || _contentStatus === 'unknown');
    if (_showStatus) {
        var statusEl = document.createElement('p');
        statusEl.style.cssText = 'margin-bottom:1.5rem;color:#f38ba8;font-size:1.05rem;font-weight:600;padding:0.75rem 1rem;background:rgba(243,139,168,0.1);border-left:3px solid #f38ba8;border-radius:4px;';
        var _statusText = content.reason || content.path_failure_reason || content.path_status_description || '';
        statusEl.textContent = _statusText || ('Path: ' + (content.source || '') + ' → ' + (content.destination || ''));
        wrap.appendChild(statusEl);
    }
    if (content.firewall_denied_by || content.policy_details) {
        var denyBlock = document.createElement('div');
        denyBlock.className = 'path-deny-details';
        denyBlock.style.marginTop = '0.5rem';
        denyBlock.style.marginBottom = '1rem';
        denyBlock.style.fontSize = '0.9rem';
        denyBlock.style.padding = '0.75rem 1rem';
        denyBlock.style.background = 'rgba(243,139,168,0.1)';
        denyBlock.style.borderLeft = '3px solid #f38ba8';
        denyBlock.style.borderRadius = '4px';
        if (content.firewall_denied_by) {
            var fwLine = document.createElement('p');
            fwLine.style.margin = '0.2rem 0';
            fwLine.style.color = '#f38ba8';
            fwLine.style.fontWeight = '600';
            fwLine.textContent = 'Denied by firewall: ' + content.firewall_denied_by;
            denyBlock.appendChild(fwLine);
        }
        if (content.policy_details) {
            var polLine = document.createElement('p');
            polLine.style.margin = '0.2rem 0';
            polLine.style.color = '#a6adc8';
            polLine.textContent = 'Policy: ' + content.policy_details;
            denyBlock.appendChild(polLine);
        }
        wrap.appendChild(denyBlock);
    }
    // Expand button
    var expandBtn = document.createElement('button');
    expandBtn.className = 'path-expand-btn';
    expandBtn.innerHTML = '&#x26F6; Expand';
    expandBtn.title = 'Open fullscreen view';
    expandBtn.addEventListener('click', function() {
        openPathFullscreen(wrap);
    });
    wrap.appendChild(expandBtn);

    var rowWrap = document.createElement('div');
    rowWrap.className = 'path-visual-row-wrap';
    var row = document.createElement('div');
    row.className = 'path-horizontal';
    var sourceIp = content.source || '';
    var destIp = content.destination || '';

    var nodes = [];
    var h0 = hops[0];
    console.log('DEBUG: First hop:', h0);
    nodes.push({ name: h0.from_device, type: h0.from_device_type, in: null, out: h0.out_interface, isSource: true, in_zone: null, out_zone: h0.out_zone, dg: h0.device_group });

    // Filter out hops with null to_device and duplicate hops (malformed API data)
    var validHops = [];
    var seenDevices = new Set([h0.from_device]);
    for (var i = 0; i < hops.length; i++) {
        console.log('DEBUG: Hop ' + i + ':', hops[i]);
        // Skip if to_device is null or already seen (duplicate)
        if (hops[i].to_device && !seenDevices.has(hops[i].to_device)) {
            validHops.push(hops[i]);
            seenDevices.add(hops[i].to_device);
        }
    }

    for (var i = 0; i < validHops.length; i++) {
        var hop = validHops[i];
        var inInt = hop.in_interface;
        var outInt = hop.out_interface;
        nodes.push({
            name: hop.to_device,
            type: hop.to_device_type,
            in: inInt,
            out: outInt,
            isDest: i === validHops.length - 1,
            in_zone: hop.in_zone,
            out_zone: hop.out_zone,
            dg: hop.device_group
        });
    }
    console.log('DEBUG: Valid hops:', validHops);
    console.log('DEBUG: All nodes:', nodes);

    for (var n = 0; n < nodes.length; n++) {
        var node = nodes[n];
        var inInt = node.in != null ? normalizeInterface(node.in) : '';
        var outInt = node.out != null ? normalizeInterface(node.out) : '';
        var item = document.createElement('div');
        item.className = 'path-item';
        if (node.isSource) {
            var badgeA = document.createElement('span');
            badgeA.className = 'path-badge source';
            badgeA.textContent = 'A';
            item.appendChild(badgeA);
        } else if (node.isDest) {
            var badgeB = document.createElement('span');
            badgeB.className = 'path-badge dest';
            badgeB.textContent = 'B';
            item.appendChild(badgeB);
        }
        appendPathIconOrFallback(item, node.type);
        var body = document.createElement('div');
        body.className = 'path-node-body';
        var nameEl = document.createElement('div');
        nameEl.className = 'path-device-name';
        var displayName = (node.name && node.name !== 'Unknown') ? node.name : null;
        if (!displayName && node.isDest && destIp) displayName = destIp;
        if (!displayName && node.isSource && sourceIp) displayName = sourceIp;
        nameEl.textContent = displayName || 'Device';
        body.appendChild(nameEl);
        if (node.isSource && sourceIp) {
            var ipEl = document.createElement('div');
            ipEl.className = 'path-ip';
            ipEl.textContent = sourceIp;
            body.appendChild(ipEl);
        }
        if (node.isDest && destIp) {
            var ipEl2 = document.createElement('div');
            ipEl2.className = 'path-ip';
            ipEl2.textContent = destIp;
            body.appendChild(ipEl2);
        }
        var intParts = [];
        if (inInt) intParts.push('In: ' + inInt);
        if (outInt) intParts.push('Out: ' + outInt);
        if (intParts.length) {
            var intEl = document.createElement('div');
            intEl.className = 'path-interfaces';
            intEl.textContent = intParts.join(' | ');
            body.appendChild(intEl);
        }
        var typeStr = (node.type || '').toLowerCase();
        var isFirewall = typeStr.indexOf('firewall') !== -1 || typeStr.indexOf('fw') !== -1 || typeStr.indexOf('palo') !== -1;
        if (isFirewall && (node.in_zone || node.out_zone || node.dg)) {
            var zParts = [];
            if (node.in_zone) zParts.push('In: ' + node.in_zone);
            if (node.out_zone) zParts.push('Out: ' + node.out_zone);
            if (node.dg) zParts.push('DG: ' + node.dg);
            if (zParts.length) {
                var zEl = document.createElement('div');
                zEl.className = 'path-zones';
                zEl.textContent = zParts.join(' | ');
                body.appendChild(zEl);
            }
        }
        item.appendChild(body);
        row.appendChild(item);
    }
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'path-connectors-svg');
    svg.setAttribute('aria-hidden', 'true');
    row.insertBefore(svg, row.firstChild);
    rowWrap.appendChild(row);
    wrap.appendChild(rowWrap);

    // Add firewall details table if there are any firewalls in the path
    var firewalls = nodes.filter(function(node) {
        var typeStr = (node.type || '').toLowerCase();
        return typeStr.indexOf('firewall') !== -1 || typeStr.indexOf('fw') !== -1 || typeStr.indexOf('palo') !== -1;
    });

    if (firewalls.length > 0) {
        var fwSection = document.createElement('div');
        fwSection.style.marginTop = '2rem';
        fwSection.style.padding = '1.5rem';
        fwSection.style.background = 'rgba(30, 30, 46, 0.4)';
        fwSection.style.borderRadius = '8px';
        fwSection.style.border = '1px solid rgba(137, 180, 250, 0.15)';
        fwSection.style.overflowX = 'auto';
        fwSection.style.overflowY = 'hidden';

        var fwToolbar = document.createElement('div');
        fwToolbar.style.display = 'flex';
        fwToolbar.style.justifyContent = 'space-between';
        fwToolbar.style.alignItems = 'center';
        fwToolbar.style.marginBottom = '1rem';

        var fwTitle = document.createElement('h4');
        fwTitle.textContent = 'Firewall Details';
        fwTitle.style.margin = '0';
        fwTitle.style.color = '#cba6f7';
        fwTitle.style.fontSize = '1rem';
        fwTitle.style.fontWeight = '600';
        fwToolbar.appendChild(fwTitle);

        var fwExportBtn = document.createElement('button');
        fwExportBtn.type = 'button';
        fwExportBtn.className = 'export-csv';
        fwExportBtn.textContent = 'Export CSV';
        fwExportBtn.style.padding = '0.4rem 0.875rem';
        fwExportBtn.style.background = 'rgba(137, 180, 250, 0.15)';
        fwExportBtn.style.color = '#89b4fa';
        fwExportBtn.style.border = '1px solid rgba(137, 180, 250, 0.3)';
        fwExportBtn.style.borderRadius = '8px';
        fwExportBtn.style.cursor = 'pointer';
        fwExportBtn.style.fontSize = '0.8rem';
        fwExportBtn.style.fontWeight = '500';
        fwExportBtn.style.transition = 'all 0.2s';
        fwExportBtn.addEventListener('mouseenter', function() { this.style.background = 'rgba(137, 180, 250, 0.25)'; });
        fwExportBtn.addEventListener('mouseleave', function() { this.style.background = 'rgba(137, 180, 250, 0.15)'; });
        fwToolbar.appendChild(fwExportBtn);

        fwSection.appendChild(fwToolbar);

        var fwTable = document.createElement('table');
        fwTable.style.width = '100%';
        fwTable.style.borderCollapse = 'separate';
        fwTable.style.borderSpacing = '0';
        fwTable.style.fontSize = '0.9rem';
        fwTable.style.borderRadius = '8px';
        fwTable.style.overflow = 'hidden';
        fwTable.style.background = 'rgba(30, 30, 46, 0.5)';

        var thead = document.createElement('thead');
        var headerRow = document.createElement('tr');
        ['Device', 'In Interface', 'Out Interface', 'In Zone', 'Out Zone', 'Device Group'].forEach(function(text) {
            var th = document.createElement('th');
            th.textContent = text;
            th.style.textAlign = 'left';
            th.style.padding = '0.875rem 1rem';
            th.style.borderBottom = '2px solid rgba(137, 180, 250, 0.25)';
            th.style.background = 'rgba(137, 180, 250, 0.12)';
            th.style.color = '#89b4fa';
            th.style.fontWeight = '600';
            th.style.textTransform = 'uppercase';
            th.style.fontSize = '0.7rem';
            th.style.letterSpacing = '0.08em';
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        fwTable.appendChild(thead);

        var tbody = document.createElement('tbody');
        firewalls.forEach(function(fw, idx) {
            var row = document.createElement('tr');
            row.style.borderBottom = '1px solid rgba(69, 71, 90, 0.3)';
            row.style.transition = 'all 0.2s ease';
            if (idx % 2 === 0) row.style.background = 'rgba(49, 50, 68, 0.2)';

            // Add hover effect
            row.addEventListener('mouseenter', function() {
                this.style.background = 'rgba(137, 180, 250, 0.15)';
            });
            row.addEventListener('mouseleave', function() {
                this.style.background = idx % 2 === 0 ? 'rgba(49, 50, 68, 0.2)' : 'transparent';
            });

            [fw.name || 'Unknown',
             fw.in || '—',
             fw.out || '—',
             fw.in_zone || '—',
             fw.out_zone || '—',
             fw.dg || '—'].forEach(function(value) {
                var td = document.createElement('td');
                td.textContent = value;
                td.style.padding = '0.875rem 1rem';
                td.style.color = '#cdd6f4';
                td.style.fontSize = '0.9rem';
                row.appendChild(td);
            });

            // Remove border from last row
            if (idx === firewalls.length - 1) {
                row.style.borderBottom = 'none';
            }

            tbody.appendChild(row);
        });
        fwTable.appendChild(tbody);
        fwSection.appendChild(fwTable);

        // CSV export for Firewall Details
        fwExportBtn.addEventListener('click', function() {
            var headers = ['Device', 'In Interface', 'Out Interface', 'In Zone', 'Out Zone', 'Device Group'];
            function esc(v) { var s = String(v == null ? '' : v).trim(); if (/[,\r\n"]/.test(s)) return '"' + s.replace(/"/g, '""') + '"'; return s; }
            var lines = [headers.map(esc).join(',')];
            firewalls.forEach(function(fw) {
                lines.push([fw.name || '', fw.in || '', fw.out || '', fw.in_zone || '', fw.out_zone || '', fw.dg || ''].map(esc).join(','));
            });
            var blob = new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'firewall_details.csv';
            a.click();
            URL.revokeObjectURL(url);
        });

        wrap.appendChild(fwSection);
    }

    container.appendChild(wrap);
    requestAnimationFrame(function() { drawPathConnectors(rowWrap); });
    if (typeof ResizeObserver !== 'undefined') {
        var ro = new ResizeObserver(function() { drawPathConnectors(rowWrap); });
        ro.observe(rowWrap);
    }
}

function drawPathConnectors(rowWrap) {
    var row = rowWrap.querySelector('.path-horizontal');
    var items = row ? row.querySelectorAll('.path-item') : [];
    var svg = row ? row.querySelector('.path-connectors-svg') : null;
    if (!row || !svg || items.length < 2) return;

    var w = row.offsetWidth;
    var h = row.offsetHeight;
    svg.setAttribute('width', w);
    svg.setAttribute('height', h);
    svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    svg.innerHTML = '';

    // Add arrow marker definition
    var defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    var marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    marker.setAttribute('id', 'arrowhead');
    marker.setAttribute('markerWidth', '10');
    marker.setAttribute('markerHeight', '10');
    marker.setAttribute('refX', '9');
    marker.setAttribute('refY', '3');
    marker.setAttribute('orient', 'auto');
    var polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    polygon.setAttribute('points', '0 0, 10 3, 0 6');
    polygon.setAttribute('fill', '#89b4fa');
    marker.appendChild(polygon);
    defs.appendChild(marker);
    svg.appendChild(defs);

    // Draw connection lines between devices
    for (var i = 0; i < items.length - 1; i++) {
        var a = items[i];
        var b = items[i + 1];
        var x1 = a.offsetLeft + a.offsetWidth;
        var y1 = a.offsetTop + a.offsetHeight / 2;
        var x2 = b.offsetLeft;
        var y2 = b.offsetTop + b.offsetHeight / 2;

        // Draw straight line with arrow
        var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', '#89b4fa');
        line.setAttribute('stroke-width', '2.5');
        line.setAttribute('stroke-linecap', 'round');
        line.setAttribute('marker-end', 'url(#arrowhead)');
        line.style.opacity = '0.8';
        svg.appendChild(line);
    }
}

function cellText(val) {
    if (val == null) return '';
    if (typeof val === 'string' && val.trim().charAt(0) === '{') {
        try {
            var obj = JSON.parse(val);
            if (obj && typeof obj === 'object') {
                var name = (obj.intfDisplaySchemaObj && obj.intfDisplaySchemaObj.value) || obj.PhysicalInftName || obj.name || obj.value;
                if (name != null) return String(name);
            }
        } catch (e) {}
    }
    if (Array.isArray(val)) {
        if (val.length === 0) return '—';
        if (val.every(function(x) { return x == null || typeof x !== 'object'; }))
            return val.map(function(x) { return x == null ? '' : String(x); }).join(', ');
        return val.map(function(item) {
            if (item == null) return '';
            if (typeof item !== 'object') return String(item);
            if (item.name != null && item.value != null) return item.name + ' (' + item.value + ')';
            if (item.name != null) return item.name;
            var parts = [];
            for (var k in item) if (Object.prototype.hasOwnProperty.call(item, k) && item[k] != null && typeof item[k] !== 'object') parts.push(item[k]);
            return parts.length ? parts.join(', ') : JSON.stringify(item);
        }).join('; ');
    }
    if (typeof val === 'object') {
        var keys = Object.keys(val).filter(function(k) { return val[k] != null && typeof val[k] !== 'object'; });
        if (keys.length <= 3) return keys.map(function(k) { return val[k]; }).join(', ');
        return keys.slice(0, 3).map(function(k) { return k + ': ' + val[k]; }).join('; ');
    }
    return String(val);
}

function isArrayOfObjects(val) {
    return Array.isArray(val) && val.length > 0 && val.every(function(x) { return x != null && typeof x === 'object' && !Array.isArray(x); });
}
var PANORAMA_COLUMN_ORDER = {
    address_objects: ["name", "type", "value", "location", "device_group"],
    address_groups: ["name", "contains_address_object", "members", "location", "device_group"],
    members: ["name", "type", "value", "location", "device_group"],
    policies: ["name", "type", "rulebase", "action", "source", "destination", "services", "address_groups", "address_objects", "location", "device_group"]
};
var PANORAMA_TABLE_LABELS = { address_objects: "Address objects", address_groups: "Address groups", members: "Address group members (IPs)", policies: "Policy details" };
var DEVICE_RACK_KEYS = ['device', 'rack', 'position', 'face', 'site', 'status', 'device_type'];
function isDeviceRackRow(row) {
    if (!row || typeof row !== 'object') return false;
    var has = 0;
    for (var i = 0; i < DEVICE_RACK_KEYS.length; i++) if (Object.prototype.hasOwnProperty.call(row, DEVICE_RACK_KEYS[i])) has++;
    return has >= 4;
}
function verticalTableFromRow(row, preferredKeys) {
    var keys = preferredKeys && preferredKeys.length
        ? preferredKeys.filter(function(k) { return Object.prototype.hasOwnProperty.call(row, k); }).concat(Object.keys(row).filter(function(k) { return preferredKeys.indexOf(k) === -1 && k.indexOf('_debug') !== 0 && k.indexOf('ai_analysis') !== 0; }))
        : Object.keys(row).filter(function(k) { return k.indexOf('_debug') !== 0 && k.indexOf('ai_analysis') !== 0; });
    var table = document.createElement('table');
    table.className = 'vertical-key-value-table';
    var tbody = document.createElement('tbody');
    keys.forEach(function(k) {
        var tr = document.createElement('tr');
        var th = document.createElement('th');
        th.textContent = k.replace(/_/g, ' ');
        var td = document.createElement('td');
        td.textContent = cellText(row[k]);
        tr.appendChild(th);
        tr.appendChild(td);
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}
function tableFromRows(rows, preferredKeys) {
    if (!rows.length) return null;
    var first = rows[0];
    var keys = preferredKeys && preferredKeys.length
        ? preferredKeys.filter(function(k) { return Object.prototype.hasOwnProperty.call(first, k); }).concat(Object.keys(first).filter(function(k) { return preferredKeys.indexOf(k) === -1; }))
        : Object.keys(first);
    var table = document.createElement('table');
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    keys.forEach(function(k) { var th = document.createElement('th'); th.textContent = k.replace(/_/g, ' '); headRow.appendChild(th); });
    thead.appendChild(headRow);
    table.appendChild(thead);
    var tbody = document.createElement('tbody');
    rows.forEach(function(row) {
        var tr = document.createElement('tr');
        keys.forEach(function(k) { var td = document.createElement('td'); td.textContent = cellText(row[k]); tr.appendChild(td); });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}
function wrapTableWithPaginationAndFilter(table) {
    var tbody = table.querySelector('tbody');
    if (!tbody) return table;
    var rows = Array.from(tbody.querySelectorAll('tr'));
    if (rows.length === 0) return table;
    var wrap = document.createElement('div');
    wrap.className = 'table-with-controls';
    var toolbar = document.createElement('div');
    toolbar.className = 'table-toolbar';
    var filterInput = document.createElement('input');
    filterInput.type = 'text';
    filterInput.placeholder = 'Filter table...';
    toolbar.appendChild(filterInput);
    var exportBtn = document.createElement('button');
    exportBtn.type = 'button';
    exportBtn.className = 'export-csv';
    exportBtn.textContent = 'Export CSV';
    toolbar.appendChild(exportBtn);
    var paginationDiv = document.createElement('div');
    paginationDiv.className = 'table-pagination';
    var infoSpan = document.createElement('span');
    var prevBtn = document.createElement('button');
    prevBtn.textContent = 'Previous';
    var nextBtn = document.createElement('button');
    nextBtn.textContent = 'Next';
    var pageSize = 10;
    var currentPage = 1;
    function updateDisplay() {
        var filterVal = (filterInput.value || '').toLowerCase().trim();
        var filtered = filterVal ? rows.filter(function(tr) { return tr.textContent.toLowerCase().includes(filterVal); }) : rows;
        var total = filtered.length;
        var totalPages = Math.max(1, Math.ceil(total / pageSize));
        currentPage = Math.min(Math.max(1, currentPage), totalPages);
        var start = (currentPage - 1) * pageSize;
        var end = Math.min(start + pageSize, total);
        rows.forEach(function(tr) { tr.style.display = 'none'; });
        filtered.slice(start, end).forEach(function(tr) { tr.style.display = ''; });
        infoSpan.textContent = total === 0 ? 'No rows' : total <= pageSize && !filterVal ? 'Showing all ' + total : 'Showing ' + (start + 1) + '–' + end + ' of ' + total;
        prevBtn.disabled = currentPage <= 1;
        nextBtn.disabled = currentPage >= totalPages;
    }
    filterInput.addEventListener('input', function() { currentPage = 1; updateDisplay(); });
    prevBtn.addEventListener('click', function() { currentPage--; updateDisplay(); });
    nextBtn.addEventListener('click', function() { currentPage++; updateDisplay(); });
    exportBtn.addEventListener('click', function() {
        var thead = table.querySelector('thead');
        var tbody = table.querySelector('tbody');
        if (!thead || !tbody) return;
        var headerCells = thead.querySelectorAll('th');
        var headers = Array.from(headerCells).map(function(th) { return th.textContent.trim(); });
        function escapeCsv(val) {
            var s = String(val == null ? '' : val).trim();
            if (/[,\r\n"]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
            return s;
        }
        var lines = [ headers.map(escapeCsv).join(',') ];
        Array.from(tbody.querySelectorAll('tr')).forEach(function(tr) {
            var cells = tr.querySelectorAll('td');
            var row = Array.from(cells).map(function(td) { return escapeCsv(td.textContent); });
            if (row.length) lines.push(row.join(','));
        });
        var csv = lines.join('\r\n');
        var blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'export.csv';
        a.click();
        URL.revokeObjectURL(url);
    });
    var tableParent = table.parentNode;
    var tableNext = table.nextSibling;
    wrap.appendChild(toolbar);
    wrap.appendChild(table);
    paginationDiv.appendChild(infoSpan);
    paginationDiv.appendChild(prevBtn);
    paginationDiv.appendChild(nextBtn);
    wrap.appendChild(paginationDiv);
    updateDisplay();
    if (tableParent) tableParent.insertBefore(wrap, tableNext);
    return wrap;
}

function tryAppendObjectAsTables(container, data) {
    if (data == null || typeof data !== 'object') return false;
    if (Array.isArray(data) && isArrayOfObjects(data)) {
        if (data.length === 1 && isDeviceRackRow(data[0])) {
            var vTable = verticalTableFromRow(data[0], DEVICE_RACK_KEYS);
            if (vTable) { container.appendChild(vTable); return true; }
        }
        var table = tableFromRows(data);
        if (table) { container.appendChild(table); wrapTableWithPaginationAndFilter(table); return true; }
    }
    if (!Array.isArray(data) && isDeviceRackRow(data)) {
        var vTable = verticalTableFromRow(data, DEVICE_RACK_KEYS);
        if (vTable) { container.appendChild(vTable); return true; }
    }
    var arrayKeys = Object.keys(data).filter(function(k) {
        var v = data[k];
        return Array.isArray(v) && v.length > 0 && v.every(function(x) { return x != null && typeof x === 'object' && !Array.isArray(x); });
    });
    var flatKeys = Object.keys(data).filter(function(k) {
        if (arrayKeys.indexOf(k) >= 0 || k === 'ai_analysis') return false;
        var v = data[k];
        return v != null && typeof v !== 'object' && (typeof v !== 'string' || v.length <= 500);
    });
    if (arrayKeys.length > 0 || flatKeys.length > 0) {
        var showFlat = flatKeys.length > 0 && flatKeys.length <= 25;
        if (showFlat && arrayKeys.length > 0) {
            var flatLower = flatKeys.map(function(k) { return k.toLowerCase(); });
            var hasRack = flatLower.indexOf('site') >= 0 || flatLower.indexOf('facility') >= 0 || (arrayKeys.indexOf('devices') >= 0 && flatLower.some(function(k) { return k === 'name' || k === 'rack_name'; }));
            var looksPanorama = (flatLower.indexOf('ip_address') >= 0 || flatLower.indexOf('vsys') >= 0) && (arrayKeys.indexOf('address_objects') >= 0 || arrayKeys.indexOf('address_groups') >= 0);
            if (!hasRack || looksPanorama) showFlat = false;
        }
        if (showFlat) {
            if (arrayKeys.length > 0) {
                var title = document.createElement('p');
                title.className = 'summary-heading';
                title.textContent = 'Rack details';
                container.appendChild(title);
            }
            var flatTable = tableFromRows([Object.fromEntries(flatKeys.map(function(k) { return [k, data[k]]; }))]);
            if (flatTable) { container.appendChild(flatTable); wrapTableWithPaginationAndFilter(flatTable); }
        }
        var isPanorama = arrayKeys.indexOf('address_objects') >= 0 || arrayKeys.indexOf('address_groups') >= 0 || arrayKeys.indexOf('policies') >= 0 || arrayKeys.indexOf('members') >= 0;
        var tableOrder = isPanorama ? ['members', 'address_objects', 'address_groups', 'policies'] : arrayKeys;
        tableOrder.forEach(function(key) {
            if (arrayKeys.indexOf(key) < 0 || key === 'path_hops') return;
            var heading = document.createElement('p');
            heading.className = 'summary-heading';
            heading.textContent = (PANORAMA_TABLE_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); }));
            container.appendChild(heading);
            var colOrder = PANORAMA_COLUMN_ORDER[key] || null;
            var arr = data[key];
            if (arr.length === 1 && isDeviceRackRow(arr[0])) {
                var vTable = verticalTableFromRow(arr[0], DEVICE_RACK_KEYS);
                if (vTable) { container.appendChild(vTable); }
            } else {
                var table = tableFromRows(data[key], colOrder);
                if (table) { container.appendChild(table); wrapTableWithPaginationAndFilter(table); }
            }
        });
        return true;
    }
    var scalarKeys = Object.keys(data).filter(function(k) { return data[k] != null && typeof data[k] !== 'object'; });
    if (scalarKeys.length > 0 && scalarKeys.length <= 20) {
        var st = tableFromRows([Object.fromEntries(scalarKeys.map(function(k) { return [k, data[k]]; }))]);
        if (st) { container.appendChild(st); wrapTableWithPaginationAndFilter(st); return true; }
    }
    return false;
}

function openPathFullscreen(pathVisualEl) {
    var overlay = document.createElement('div');
    overlay.className = 'path-fullscreen-overlay';

    // Header bar
    var header = document.createElement('div');
    header.className = 'path-fullscreen-header';
    var title = document.createElement('span');
    title.className = 'path-fullscreen-title';
    title.textContent = 'Network Path';
    var closeBtn = document.createElement('button');
    closeBtn.className = 'path-fullscreen-close';
    closeBtn.textContent = 'Close (Esc)';
    var cleanup = function() { overlay.remove(); document.removeEventListener('keydown', escHandler); };
    closeBtn.addEventListener('click', cleanup);
    header.appendChild(title);
    header.appendChild(closeBtn);
    overlay.appendChild(header);

    // Body with cloned path visual
    var body = document.createElement('div');
    body.className = 'path-fullscreen-body';
    var clone = pathVisualEl.cloneNode(true);
    clone.className = 'path-visual-clone';
    clone.style.background = 'none';
    clone.style.border = 'none';
    clone.style.margin = '0';
    clone.style.maxWidth = 'none';
    clone.style.overflow = 'visible';
    // Remove the expand button from the clone
    var cloneBtn = clone.querySelector('.path-expand-btn');
    if (cloneBtn) cloneBtn.remove();
    body.appendChild(clone);
    overlay.appendChild(body);

    // Close on Escape key
    var escHandler = function(e) {
        if (e.key === 'Escape') cleanup();
    };
    document.addEventListener('keydown', escHandler);

    // Close on clicking backdrop (not the content)
    overlay.addEventListener('click', function(e) {
        if (e.target === overlay || e.target === body) cleanup();
    });

    document.body.appendChild(overlay);

    // Redraw SVG connectors in the cloned view after layout
    requestAnimationFrame(function() {
        var cloneRowWrap = clone.querySelector('.path-visual-row-wrap');
        if (cloneRowWrap) drawPathConnectors(cloneRowWrap);
    });
}

function appendMessage(role, content) {
    var div = document.createElement('div');
    div.className = 'msg ' + role;
    if (content === undefined || content === null || (typeof content === 'string' && !content.trim())) {
        div.textContent = 'No response received.';
    } else if (typeof content === 'string') {
        div.textContent = content;
    } else if (typeof content === 'object') {
        // Handle yes/no answer first - display it prominently at the top
        if (content.yes_no_answer) {
            var yesNoDiv = document.createElement('div');
            yesNoDiv.className = 'yes-no-answer';
            yesNoDiv.style.cssText = 'font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; padding: 0.75rem 1rem; border-radius: 8px; background: linear-gradient(135deg, rgba(137, 180, 250, 0.1) 0%, rgba(203, 166, 247, 0.1) 100%); border: 1px solid rgba(137, 180, 250, 0.3);';
            yesNoDiv.textContent = content.yes_no_answer;
            div.appendChild(yesNoDiv);
        }

        // Handle specific metric answers - display them prominently
        if (content.metric_answer) {
            var metricDiv = document.createElement('div');
            metricDiv.className = 'metric-answer';
            metricDiv.style.cssText = 'font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; padding: 0.75rem 1rem; border-radius: 8px; background: linear-gradient(135deg, rgba(166, 227, 161, 0.1) 0%, rgba(137, 180, 250, 0.1) 100%); border: 1px solid rgba(166, 227, 161, 0.3);';
            metricDiv.textContent = content.metric_answer;
            div.appendChild(metricDiv);
        }

        // Handle Panorama direct answers - display them prominently
        if (content.direct_answer) {
            var directDiv = document.createElement('div');
            directDiv.className = 'direct-answer';
            directDiv.style.cssText = 'font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; padding: 0.75rem 1rem; border-radius: 8px; background: linear-gradient(135deg, rgba(249, 226, 175, 0.1) 0%, rgba(166, 227, 161, 0.1) 100%); border: 1px solid rgba(249, 226, 175, 0.3);';
            directDiv.textContent = content.direct_answer;
            div.appendChild(directDiv);
        }

        if (content.batch_results && Array.isArray(content.batch_results)) {
            renderBatchResults(div, content.batch_results, content.tool || '');
        } else if (content.error) {
            // Don't treat clarification requests as errors (no red styling)
            if (content.requires_site && content.sites && content.sites.length > 0) {
                // This is a clarification question, not an error - display normally
                var p = document.createElement('p');
                p.textContent = content.error;
                div.appendChild(p);
                var prompt = document.createElement('p');
                prompt.style.cssText = 'font-weight: 500; margin-top: 0.5rem; color: #89b4fa;';
                prompt.textContent = 'Which site? ' + content.sites.join(', ') + '. Reply with the site name.';
                div.appendChild(prompt);
            } else {
                // Actual error - show in red
                div.classList.add('err');
                div.textContent = content.error;
            }
        } else if (content.path_hops && Array.isArray(content.path_hops) && content.path_hops.length > 0) {
            renderPathWithIcons(div, content);
            /* No extra tables after path visual - path is the main content */
        } else if (content.source && content.destination && !content.error && (!content.path_hops || content.path_hops.length === 0)) {
            /* Path was calculated but hop details not available - show compact summary */
            var pathWrap = document.createElement('div');
            pathWrap.className = 'path-visual';
            var pathLine = document.createElement('p');
            pathLine.className = 'path-status';
            pathLine.textContent = 'Path: ' + content.source + ' → ' + content.destination;
            pathWrap.appendChild(pathLine);
            var statusDesc = content.path_status_description || content.path_status || content.statusDescription || '';
            var _descNoise = ['l2 connections has not been discovered', 'l2 connection has not been discovered'];
            var _descIsNoise = _descNoise.some(function(p) { return statusDesc.toLowerCase().indexOf(p) !== -1; });
            if (statusDesc && !_descIsNoise) {
                var statusP = document.createElement('p');
                statusP.className = 'path-status';
                statusP.style.marginTop = '0.5rem';
                statusP.textContent = statusDesc;
                if ((content.path_status || content.statusCode || '').toString().toLowerCase().indexOf('fail') !== -1) statusP.classList.add('failed');
                pathWrap.appendChild(statusP);
            }
            if (content.message) {
                var msgP = document.createElement('p');
                msgP.style.marginTop = '0.5rem'; msgP.style.fontSize = '0.9rem'; msgP.style.color = '#a6adc8';
                msgP.textContent = content.message;
                pathWrap.appendChild(msgP);
            } else if (content.note) {
                var noteP = document.createElement('p');
                noteP.style.marginTop = '0.5rem'; noteP.style.fontSize = '0.9rem'; noteP.style.color = '#a6adc8';
                noteP.textContent = content.note;
                pathWrap.appendChild(noteP);
            }
            if ((content.status === 'denied' || (content.reason && content.status === 'denied')) && (content.firewall_denied_by || content.policy_details)) {
                if (content.firewall_denied_by) {
                    var fwP = document.createElement('p');
                    fwP.style.marginTop = '0.5rem'; fwP.style.color = '#f38ba8'; fwP.style.fontSize = '0.9rem';
                    fwP.textContent = 'Denied by firewall: ' + content.firewall_denied_by;
                    pathWrap.appendChild(fwP);
                }
                if (content.policy_details) {
                    var polP = document.createElement('p');
                    polP.style.marginTop = '0.25rem'; polP.style.color = '#a6adc8'; polP.style.fontSize = '0.9rem';
                    polP.textContent = 'Policy: ' + content.policy_details;
                    pathWrap.appendChild(polP);
                }
            }
            div.appendChild(pathWrap);
        } else if (content.message || (content.ai_analysis && (content.ai_analysis.summary || content.ai_analysis.Summary))) {
            var isDeviceRack = isDeviceRackRow(content);
            var summaryText = content.message || (content.ai_analysis && (content.ai_analysis.summary || content.ai_analysis.Summary));
            var sumStr = summaryText && (typeof summaryText === 'string' ? summaryText : JSON.stringify(summaryText));
            var hasStructuredTables = (Array.isArray(content.address_objects) && content.address_objects.length > 0) ||
                (Array.isArray(content.address_groups) && content.address_groups.length > 0) ||
                (Array.isArray(content.policies) && content.policies.length > 0);
            var looksLikeMarkdownTable = sumStr && (/^\s*\|[\s\-:]+\|/.test(sumStr) || /\*\*Table\s*\d/i.test(sumStr) || sumStr.indexOf('| ---') !== -1);
            var showSummary = sumStr && (!hasStructuredTables || (!looksLikeMarkdownTable && sumStr.length <= 500));
            if (isDeviceRack && showSummary) {
                tryAppendObjectAsTables(div, content);
                var gap1 = document.createElement('p');
                gap1.style.marginTop = '1rem'; gap1.style.marginBottom = '0.5rem'; gap1.innerHTML = '&nbsp;';
                div.appendChild(gap1);
                var sp = document.createElement('p');
                sp.textContent = sumStr;
                sp.style.marginBottom = '0.75rem';
                div.appendChild(sp);
            } else {
                if (content.message) {
                    var mp = document.createElement('p');
                    mp.textContent = content.message;
                    mp.style.marginBottom = '0.75rem';
                    div.appendChild(mp);
                }
                if (content.ai_analysis && (content.ai_analysis.summary || content.ai_analysis.Summary) && showSummary) {
                    var sp = document.createElement('p');
                    sp.textContent = sumStr;
                    sp.style.marginBottom = '0.75rem';
                    div.appendChild(sp);
                }
                tryAppendObjectAsTables(div, content);
            }
        } else if (tryAppendObjectAsTables(div, content)) {
        } else {
            var pre = document.createElement('pre');
            pre.textContent = JSON.stringify(content, null, 2);
            div.appendChild(pre);
        }
    } else {
        div.textContent = String(content);
    }
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderBatchResults(container, results, tool) {
    if (!Array.isArray(results) || results.length === 0) {
        container.textContent = 'No results returned.';
        return;
    }
    var isPathQuery = tool === 'query_network_path';

    // Summary stats
    var counts = { allowed: 0, denied: 0, unknown: 0, error: 0, success: 0, failed: 0 };
    results.forEach(function(r) {
        var s = (r.status || '').toLowerCase();
        if (counts.hasOwnProperty(s)) counts[s]++;
        else counts.unknown++;
    });

    var summary = document.createElement('div');
    summary.className = 'batch-summary';
    if (isPathQuery) {
        summary.innerHTML =
            '<span class="batch-stat allowed">' + counts.success + ' Success</span>' +
            '<span class="batch-stat denied">' + counts.failed + ' Failed</span>' +
            (counts.error > 0 ? '<span class="batch-stat error">' + counts.error + ' Error</span>' : '') +
            (counts.unknown > 0 ? '<span class="batch-stat unknown">' + counts.unknown + ' Unknown</span>' : '') +
            '<span style="color:#a6adc8;"> out of ' + results.length + ' total</span>';
    } else {
        summary.innerHTML =
            '<span class="batch-stat allowed">' + counts.allowed + ' Allowed</span>' +
            '<span class="batch-stat denied">' + counts.denied + ' Denied</span>' +
            '<span class="batch-stat unknown">' + counts.unknown + ' Unknown</span>' +
            (counts.error > 0 ? '<span class="batch-stat error">' + counts.error + ' Error</span>' : '') +
            '<span style="color:#a6adc8;"> out of ' + results.length + ' total</span>';
    }
    container.appendChild(summary);

    // Table columns differ by tool
    var headers = isPathQuery
        ? ['Source', 'Destination', 'Protocol', 'Port', 'Status', 'Path', 'Details']
        : ['Source', 'Destination', 'Protocol', 'Port', 'Status', 'Reason', 'Firewall', 'Policy'];

    var table = document.createElement('table');
    table.className = 'batch-table';
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    headers.forEach(function(h) {
        var th = document.createElement('th');
        th.textContent = h;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    results.forEach(function(r) {
        var tr = document.createElement('tr');
        var statusLower = (r.status || 'unknown').toLowerCase();

        // Common columns: Source, Destination, Protocol, Port
        ['source', 'destination'].forEach(function(key) {
            var td = document.createElement('td');
            td.textContent = r[key] || '';
            tr.appendChild(td);
        });
        var tdProto = document.createElement('td');
        tdProto.textContent = (r.protocol || 'tcp').toUpperCase();
        tr.appendChild(tdProto);
        var tdPort = document.createElement('td');
        tdPort.textContent = r.port || '0';
        tr.appendChild(tdPort);

        // Status (color-coded)
        var tdStatus = document.createElement('td');
        var statusClass = statusLower;
        if (statusLower === 'success') statusClass = 'allowed';
        if (statusLower === 'failed') statusClass = 'denied';
        tdStatus.className = 'status-cell ' + statusClass;
        tdStatus.textContent = statusLower;
        tr.appendChild(tdStatus);

        if (isPathQuery) {
            // Path summary
            var tdPath = document.createElement('td');
            tdPath.textContent = r.path_summary || '';
            tdPath.style.whiteSpace = 'normal';
            tdPath.style.maxWidth = '300px';
            tdPath.style.fontSize = '0.8rem';
            tr.appendChild(tdPath);
            // Details (reason/status description)
            var tdDetails = document.createElement('td');
            tdDetails.textContent = r.reason || '';
            tdDetails.style.whiteSpace = 'normal';
            tdDetails.style.maxWidth = '200px';
            tr.appendChild(tdDetails);
        } else {
            // Reason
            var tdReason = document.createElement('td');
            tdReason.textContent = r.reason || '';
            tdReason.style.whiteSpace = 'normal';
            tdReason.style.maxWidth = '200px';
            tr.appendChild(tdReason);
            // Firewall
            var tdFw = document.createElement('td');
            tdFw.textContent = r.firewall_denied_by || '';
            tr.appendChild(tdFw);
            // Policy
            var tdPol = document.createElement('td');
            tdPol.textContent = r.policy_details || '';
            tdPol.style.whiteSpace = 'normal';
            tdPol.style.maxWidth = '200px';
            tr.appendChild(tdPol);
        }

        tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    var tableWrap = document.createElement('div');
    tableWrap.style.overflowX = 'auto';
    tableWrap.appendChild(table);
    container.appendChild(tableWrap);

    // Export CSV button
    var exportBtn = document.createElement('button');
    exportBtn.type = 'button';
    exportBtn.className = 'export-csv';
    exportBtn.textContent = 'Export Results CSV';
    exportBtn.style.marginTop = '0.75rem';
    exportBtn.addEventListener('click', function() {
        function esc(v) { var s = String(v || ''); if (/[,\r\n"]/.test(s)) return '"' + s.replace(/"/g, '""') + '"'; return s; }
        var csvHeaders = isPathQuery
            ? ['Source', 'Destination', 'Protocol', 'Port', 'Status', 'Path', 'Details']
            : ['Source', 'Destination', 'Protocol', 'Port', 'Status', 'Reason', 'Firewall', 'Policy'];
        var lines = [csvHeaders.map(esc).join(',')];
        results.forEach(function(r) {
            var row = isPathQuery
                ? [r.source, r.destination, r.protocol, r.port, r.status, r.path_summary, r.reason]
                : [r.source, r.destination, r.protocol, r.port, r.status, r.reason, r.firewall_denied_by, r.policy_details];
            lines.push(row.map(esc).join(','));
        });
        var blob = new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url; a.download = 'batch_results.csv'; a.click();
        URL.revokeObjectURL(url);
    });
    container.appendChild(exportBtn);

    // Render individual path graphics for each batch row (path queries only)
    if (isPathQuery) {
        results.forEach(function(r, idx) {
            if (!r.path_hops || !Array.isArray(r.path_hops) || r.path_hops.length === 0) return;
            var pathSection = document.createElement('div');
            pathSection.style.cssText = 'margin-top: 1.5rem; padding: 1.25rem; background: rgba(30, 30, 46, 0.5); border-radius: 8px; border: 1px solid rgba(137, 180, 250, 0.15);';

            var pathHeader = document.createElement('div');
            pathHeader.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;';

            var pathTitle = document.createElement('h4');
            pathTitle.style.cssText = 'margin: 0; color: #89b4fa; font-size: 0.95rem; font-weight: 600;';
            pathTitle.textContent = (r.source || '') + ' \u2192 ' + (r.destination || '');
            pathHeader.appendChild(pathTitle);

            var statusBadge = document.createElement('span');
            var sLower = (r.status || '').toLowerCase();
            statusBadge.textContent = sLower;
            statusBadge.style.cssText = 'padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;';
            if (sLower === 'success') {
                statusBadge.style.background = 'rgba(166, 227, 161, 0.15)';
                statusBadge.style.color = '#a6e3a1';
            } else {
                statusBadge.style.background = 'rgba(243, 139, 168, 0.15)';
                statusBadge.style.color = '#f38ba8';
            }
            pathHeader.appendChild(statusBadge);
            pathSection.appendChild(pathHeader);

            // Reuse renderPathWithIcons by passing a content-like object
            var fakeContent = {
                path_hops: r.path_hops,
                source: r.source,
                destination: r.destination,
                path_status: r.path_status || r.status,
                path_failure_reason: r.path_failure_reason || '',
                path_status_description: r.reason || ''
            };
            renderPathWithIcons(pathSection, fakeContent);
            container.appendChild(pathSection);
        });
    }
}

// --- File attachment flow ---
var attachedFile = null;

function onFileSelect(input) {
    if (input.files[0]) {
        attachedFile = input.files[0];
        document.getElementById('attachName').textContent = '\uD83D\uDCCE ' + attachedFile.name;
        document.getElementById('attachIndicator').style.display = 'flex';
        inputEl.placeholder = 'Describe what to do: e.g. "check if paths are allowed" or "show network path"...';
        inputEl.focus();
    }
    input.value = '';
}

function clearAttachment() {
    attachedFile = null;
    document.getElementById('attachIndicator').style.display = 'none';
    inputEl.placeholder = 'Ask about paths, devices, racks, Panorama, Splunk...';
}

async function sendBatchUpload(file, message) {
    var uploadBtn = document.getElementById('uploadBtn');
    uploadBtn.classList.add('uploading');
    sendBtn.disabled = true;

    var statusEl = document.createElement('div');
    statusEl.className = 'msg status-msg';
    statusEl.textContent = 'Processing batch upload...';
    messagesEl.appendChild(statusEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    try {
        var formData = new FormData();
        formData.append('file', file);
        formData.append('message', message);

        var res = await fetch('/api/batch-upload', {
            method: 'POST',
            body: formData,
        });
        statusEl.remove();

        var data = await res.json().catch(function() { return {}; });
        if (!res.ok) {
            appendMessage('assistant', data.error || data.detail || 'Upload failed');
        } else {
            var content = data.content || data;
            if (content.batch_results) {
                var div = document.createElement('div');
                div.className = 'msg assistant';
                renderBatchResults(div, content.batch_results, content.tool || '');
                messagesEl.appendChild(div);
                messagesEl.scrollTop = messagesEl.scrollHeight;
            } else {
                appendMessage('assistant', content);
            }
        }
    } catch (err) {
        statusEl.remove();
        var errMsg = err && err.message ? err.message : String(err);
        appendMessage('assistant', 'Upload error: ' + errMsg);
    }

    uploadBtn.classList.remove('uploading');
    sendBtn.disabled = false;
}

async function send(e) {
    e.preventDefault();
    var text = inputEl.value.trim();

    // If file is attached, send as batch upload
    if (attachedFile) {
        var file = attachedFile;
        var msg = text || 'check if paths are allowed';
        inputEl.value = '';
        clearAttachment();
        appendMessage('user', msg + '\n\uD83D\uDCCE ' + file.name);
        await sendBatchUpload(file, msg);
        return false;
    }

    if (!text) return false;
    inputEl.value = '';
    sendBtn.disabled = true;
    appendMessage('user', text);
    conversationHistory.push({ role: 'user', content: text });
    var statusEl = document.createElement('div');
    statusEl.className = 'msg status-msg';
    statusEl.textContent = 'Processing...';
    messagesEl.appendChild(statusEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    try {
        var res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, conversation_history: conversationHistory.slice(-20) })
        });
        statusEl.remove();
        var data = await res.json().catch(function() { return {}; });
        var assistantContent = !res.ok ? (data.detail || data.message || 'Request failed') : (data.content ?? data.message ?? 'No response');
        if (assistantContent === undefined || assistantContent === null) assistantContent = 'No response received.';
        appendMessage('assistant', assistantContent);
        var forHistory = typeof assistantContent === 'string' ? assistantContent : JSON.stringify(assistantContent);
        conversationHistory.push({ role: 'assistant', content: forHistory });
    } catch (err) {
        statusEl.remove();
        var errMsg = err && err.message ? err.message : String(err);
        if (errMsg.indexOf('fetch') !== -1) errMsg = 'Request failed. Check that the server is running and the request did not time out.';
        appendMessage('assistant', 'Error: ' + errMsg);
        conversationHistory.push({ role: 'assistant', content: 'Error: ' + errMsg });
    }
    sendBtn.disabled = false;
    return false;
}

// --- Health status polling ---
(function() {
    var statusEl = document.getElementById('health-status');
    if (!statusEl) return;
    var dotEl = statusEl.querySelector('.health-dot');
    var labelEl = statusEl.querySelector('.health-label');

    function update(cls, label, tooltip) {
        dotEl.className = 'health-dot ' + cls;
        labelEl.textContent = label;
        statusEl.className = 'health-status ' + cls;
        statusEl.title = tooltip;
    }

    async function poll() {
        try {
            var res = await fetch('/health', { signal: AbortSignal.timeout(5000) });
            if (!res.ok) { update('unhealthy', 'App error', 'App returned HTTP ' + res.status); return; }
            var data = await res.json();
            var mcp = data.mcp_server || 'unknown';
            var tools = data.mcp_tools_registered;
            if (mcp === 'ok' && tools > 0) {
                update('healthy', 'All systems OK', 'App: OK | MCP: OK | Tools: ' + tools);
            } else if (mcp === 'unreachable') {
                update('degraded', 'MCP offline', 'App: OK | MCP server is unreachable');
            } else {
                update('degraded', 'MCP issue', 'App: OK | MCP: ' + mcp);
            }
        } catch (e) {
            update('unhealthy', 'Offline', 'Cannot reach the server');
        }
    }

    poll();
    setInterval(poll, 30000);
})();
