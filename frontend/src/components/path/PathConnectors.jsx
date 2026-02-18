import { useRef, useEffect } from 'react'
import styles from './PathConnectors.module.css'

export default function PathConnectors({ itemCount }) {
  const svgRef = useRef(null)

  useEffect(() => {
    function draw() {
      const svg = svgRef.current
      const container = svg?.parentElement
      if (!svg || !container) return

      const items = container.querySelectorAll('.path-item')
      if (items.length < 2) return

      const w = container.offsetWidth
      const h = container.offsetHeight
      svg.setAttribute('width', w)
      svg.setAttribute('height', h)
      svg.setAttribute('viewBox', `0 0 ${w} ${h}`)
      svg.innerHTML = ''

      const ns = 'http://www.w3.org/2000/svg'
      const defs = document.createElementNS(ns, 'defs')
      const marker = document.createElementNS(ns, 'marker')
      marker.setAttribute('id', 'arrowhead')
      marker.setAttribute('markerWidth', '10')
      marker.setAttribute('markerHeight', '10')
      marker.setAttribute('refX', '9')
      marker.setAttribute('refY', '3')
      marker.setAttribute('orient', 'auto')
      const polygon = document.createElementNS(ns, 'polygon')
      polygon.setAttribute('points', '0 0, 10 3, 0 6')
      polygon.setAttribute('fill', '#58a6ff')
      marker.appendChild(polygon)
      defs.appendChild(marker)
      svg.appendChild(defs)

      for (let i = 0; i < items.length - 1; i++) {
        const a = items[i]
        const b = items[i + 1]
        const line = document.createElementNS(ns, 'line')
        line.setAttribute('x1', a.offsetLeft + a.offsetWidth)
        line.setAttribute('y1', a.offsetTop + a.offsetHeight / 2)
        line.setAttribute('x2', b.offsetLeft)
        line.setAttribute('y2', b.offsetTop + b.offsetHeight / 2)
        line.setAttribute('stroke', '#58a6ff')
        line.setAttribute('stroke-width', '2.5')
        line.setAttribute('stroke-linecap', 'round')
        line.setAttribute('marker-end', 'url(#arrowhead)')
        line.style.opacity = '0.8'
        line.style.strokeDasharray = '200'
        line.style.strokeDashoffset = '200'
        line.style.animation = 'drawLine 0.6s ease-out forwards'
        svg.appendChild(line)
      }
    }

    let ro = null
    const frame = requestAnimationFrame(draw)
    const svg = svgRef.current
    const container = svg?.parentElement
    if (container && typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(draw)
      ro.observe(container)
    }

    return () => {
      cancelAnimationFrame(frame)
      if (ro) ro.disconnect()
    }
  }, [itemCount])

  return <svg ref={svgRef} className={styles.svg} aria-hidden="true" />
}
