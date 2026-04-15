import { useState, useRef, forwardRef, useImperativeHandle } from 'react'
import { useChatStore } from '../../stores/chatStore.js'
import styles from './ChatInput.module.css'

const ChatInput = forwardRef(function ChatInput(props, ref) {
  const [inputText, setInputText] = useState('')
  const [attachedFile, setAttachedFile] = useState(null)
  const [attachName, setAttachName] = useState('')
  const fileInputRef = useRef(null)
  const isLoading = useChatStore(s => s.isLoading)
  const sendMessage = useChatStore(s => s.sendMessage)
  const sendBatch = useChatStore(s => s.sendBatch)
  const stopGeneration = useChatStore(s => s.stopGeneration)

  useImperativeHandle(ref, () => ({
    fillText: (query) => setInputText(query),
  }))

  function clearAttachment() {
    setAttachedFile(null)
    setAttachName('')
  }

  async function onSubmit(e) {
    e.preventDefault()
    if (attachedFile) {
      const file = attachedFile
      const msg = inputText.trim() || 'check if paths are allowed'
      setInputText('')
      clearAttachment()
      await sendBatch(file, msg)
      return
    }
    const text = inputText.trim()
    if (!text) return
    setInputText('')
    await sendMessage(text)
  }

  function onFileSelect(e) {
    const file = e.target.files[0]
    if (file) {
      setAttachedFile(file)
      setAttachName('\uD83D\uDCCE ' + file.name)
    }
    e.target.value = ''
  }

  return (
    <div className={styles.form}>
      {attachedFile && (
        <div className={styles.attachIndicator}>
          <span className={styles.attachName}>{attachName}</span>
          <button type="button" className={styles.attachRemove} onClick={clearAttachment} title="Remove attachment">&times;</button>
        </div>
      )}
      <form onSubmit={onSubmit} className={styles.formInner}>
        <input
          type="text"
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          placeholder={attachedFile ? 'Describe what to do: e.g. "check if paths are allowed"...' : 'Ask about paths, devices, racks, incidents, or change requests...'}
          autoComplete="off"
          className={styles.textInput}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept=".xlsx,.xls,.csv"
          style={{ display: 'none' }}
          onChange={onFileSelect}
        />
        <button
          type="button"
          className={styles.uploadBtn}
          title="Upload spreadsheet (.xlsx, .csv) for batch queries"
          onClick={() => fileInputRef.current?.click()}
        >&#128206;</button>
        {!isLoading ? (
          <button type="submit" className={styles.sendBtn} aria-label="Send">&#8593;</button>
        ) : (
          <button type="button" className={styles.stopBtn} aria-label="Stop" title="Stop generating" onClick={stopGeneration}>&#9632;</button>
        )}
      </form>
    </div>
  )
})

export default ChatInput
