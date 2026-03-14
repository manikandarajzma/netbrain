import { isArrayOfObjects, isDeviceRackRow } from './formatters.js'

/**
 * Classify an assistant response content object into a rendering type.
 * Returns { type, content } where type is one of:
 * 'text', 'batch', 'clarification', 'error', 'path', 'path-summary',
 * 'structured', 'table', 'json'
 */
export function classifyResponse(content) {
  if (content === undefined || content === null || (typeof content === 'string' && !content.trim())) {
    return { type: 'text', content: 'No response received.' }
  }
  if (typeof content === 'string') {
    return { type: 'text', content }
  }
  if (typeof content !== 'object') {
    return { type: 'text', content: String(content) }
  }

  // Multi-tool chain results
  if (content.multi_results && Array.isArray(content.multi_results)) {
    return { type: 'multi', content }
  }

  // Batch results
  if (content.batch_results && Array.isArray(content.batch_results)) {
    return { type: 'batch', content }
  }

  // Errors and clarification
  if (content.error) {
    if (content.requires_site && content.sites && content.sites.length > 0) {
      return { type: 'clarification', content }
    }
    return { type: 'error', content }
  }

  // Path with hops
  if (content.path_hops && Array.isArray(content.path_hops) && content.path_hops.length > 0) {
    return { type: 'path', content }
  }

  // Path summary (calculated but no hops)
  if (content.source && content.destination && !content.error && (!content.path_hops || content.path_hops.length === 0)) {
    return { type: 'path-summary', content }
  }

  // Structured with message or AI analysis
  if (content.message || (content.ai_analysis && (content.ai_analysis.summary || content.ai_analysis.Summary))) {
    return { type: 'structured', content }
  }

  // Try table detection
  if (Array.isArray(content) && isArrayOfObjects(content)) {
    return { type: 'table', content }
  }
  if (!Array.isArray(content) && isDeviceRackRow(content)) {
    return { type: 'table', content }
  }

  // Check for array or scalar sub-keys that could be tabled
  const arrayKeys = Object.keys(content).filter(k => {
    const v = content[k]
    return Array.isArray(v) && v.length > 0 && v.every(x => x != null && typeof x === 'object' && !Array.isArray(x))
  })
  const scalarKeys = Object.keys(content).filter(k => content[k] != null && typeof content[k] !== 'object')
  if (arrayKeys.length > 0 || (scalarKeys.length > 0 && scalarKeys.length <= 20)) {
    return { type: 'table', content }
  }

  return { type: 'json', content }
}
