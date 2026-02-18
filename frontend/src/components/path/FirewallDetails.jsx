import { downloadCsv } from '../../utils/csvExport.js'
import styles from './FirewallDetails.module.css'

export default function FirewallDetails({ firewalls }) {
  function exportCsv() {
    const headers = ['Device', 'In Interface', 'Out Interface', 'In Zone', 'Out Zone', 'Device Group']
    const rows = firewalls.map(fw => [
      fw.name || '', fw.in || '', fw.out || '', fw.in_zone || '', fw.out_zone || '', fw.dg || '',
    ])
    downloadCsv(headers, rows, 'firewall_details.csv')
  }

  return (
    <div className={styles.fwSection}>
      <div className={styles.fwToolbar}>
        <h4 className={styles.fwTitle}>Firewall Details</h4>
        <button type="button" className={styles.fwExport} onClick={exportCsv}>Export CSV</button>
      </div>
      <table className={styles.fwTable}>
        <thead>
          <tr>
            <th>Device</th>
            <th>In Interface</th>
            <th>Out Interface</th>
            <th>In Zone</th>
            <th>Out Zone</th>
            <th>Device Group</th>
          </tr>
        </thead>
        <tbody>
          {firewalls.map((fw, i) => (
            <tr key={i} className={i % 2 === 0 ? styles.even : undefined}>
              <td>{fw.name || 'Unknown'}</td>
              <td>{fw.in || '\u2014'}</td>
              <td>{fw.out || '\u2014'}</td>
              <td>{fw.in_zone || '\u2014'}</td>
              <td>{fw.out_zone || '\u2014'}</td>
              <td>{fw.dg || '\u2014'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
