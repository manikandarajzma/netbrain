export function escapeCsv(val) {
  const s = String(val == null ? '' : val).trim()
  if (/[,\r\n"]/.test(s)) return '"' + s.replace(/"/g, '""') + '"'
  return s
}

export function downloadCsv(headers, rows, filename = 'export.csv') {
  const lines = [headers.map(escapeCsv).join(',')]
  rows.forEach(row => {
    lines.push(row.map(escapeCsv).join(','))
  })
  const blob = new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
