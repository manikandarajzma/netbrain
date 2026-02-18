import { useState, useMemo } from 'react'
import { cellText } from '../../utils/formatters.js'
import { downloadCsv } from '../../utils/csvExport.js'
import styles from './DataTable.module.css'

export default function DataTable({ rows, columns = null, pageSize = 10 }) {
  const [filterVal, setFilterVal] = useState('')
  const [currentPage, setCurrentPage] = useState(1)

  const keys = useMemo(() => {
    if (!rows.length) return []
    if (columns && columns.length) {
      const first = rows[0]
      return [
        ...columns.filter(k => Object.prototype.hasOwnProperty.call(first, k)),
        ...Object.keys(first).filter(k => !columns.includes(k)),
      ]
    }
    return Object.keys(rows[0])
  }, [rows, columns])

  const filteredRows = useMemo(() => {
    const q = filterVal.toLowerCase().trim()
    if (!q) return rows
    return rows.filter(row =>
      keys.some(k => String(cellText(row[k])).toLowerCase().includes(q))
    )
  }, [rows, keys, filterVal])

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize))
  const safePage = Math.min(currentPage, totalPages)

  const pagedRows = useMemo(() => {
    const start = (safePage - 1) * pageSize
    return filteredRows.slice(start, start + pageSize)
  }, [filteredRows, safePage, pageSize])

  const pageInfo = useMemo(() => {
    const total = filteredRows.length
    if (total === 0) return 'No rows'
    if (total <= pageSize && !filterVal) return `Showing all ${total}`
    const start = (safePage - 1) * pageSize + 1
    const end = Math.min(safePage * pageSize, total)
    return `Showing ${start}\u2013${end} of ${total}`
  }, [filteredRows.length, safePage, pageSize, filterVal])

  function onFilter(e) {
    setFilterVal(e.target.value)
    setCurrentPage(1)
  }

  function exportCsv() {
    const headers = keys.map(k => k.replace(/_/g, ' '))
    const data = rows.map(row => keys.map(k => cellText(row[k])))
    downloadCsv(headers, data, 'export.csv')
  }

  function formatHeader(key) {
    return key.replace(/_/g, ' ')
  }

  return (
    <div className={styles.tableWithControls}>
      <div className={styles.tableToolbar}>
        <input value={filterVal} onChange={onFilter} type="text" placeholder="Filter table..." className={styles.filterInput} />
        <button type="button" className={styles.exportCsv} onClick={exportCsv}>Export CSV</button>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className={styles.table}>
          <thead>
            <tr>
              {keys.map(k => <th key={k}>{formatHeader(k)}</th>)}
            </tr>
          </thead>
          <tbody>
            {pagedRows.map((row, i) => (
              <tr key={i}>
                {keys.map(k => <td key={k}>{cellText(row[k])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className={styles.tablePagination}>
        <span>{pageInfo}</span>
        <button onClick={() => setCurrentPage(p => Math.max(1, p - 1))} disabled={safePage <= 1}>Previous</button>
        <button onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))} disabled={safePage >= totalPages}>Next</button>
      </div>
    </div>
  )
}
