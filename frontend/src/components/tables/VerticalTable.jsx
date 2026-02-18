import { useMemo } from 'react'
import { cellText, orderKeys } from '../../utils/formatters.js'
import styles from './VerticalTable.module.css'

export default function VerticalTable({ row, preferredKeys = null }) {
  const keys = useMemo(() => orderKeys(row, preferredKeys), [row, preferredKeys])

  return (
    <table className={styles.table}>
      <tbody>
        {keys.map(k => (
          <tr key={k}>
            <th>{k.replace(/_/g, ' ')}</th>
            <td>{cellText(row[k])}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
