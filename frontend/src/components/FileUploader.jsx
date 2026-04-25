import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'

const CLOUD_NAME = import.meta.env.VITE_CLOUDINARY_CLOUD_NAME
const UPLOAD_PRESET = import.meta.env.VITE_CLOUDINARY_UPLOAD_PRESET
const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const STAGES = {
  idle: null,
  uploading: 'Uploading to cloud ☁️…',
  processing: 'Processing document 📄…',
  done: 'Ready for queries 🚀',
  error: null,
}

export default function FileUploader() {
  const [stage, setStage] = useState('idle')
  const [filename, setFilename] = useState('')
  const [error, setError] = useState('')

  const onDrop = useCallback(async (accepted) => {
    const file = accepted[0]
    if (!file) return
    setFilename(file.name)
    setError('')
    setStage('uploading')

    try {
      const cdnForm = new FormData()
      cdnForm.append('file', file)
      cdnForm.append('upload_preset', UPLOAD_PRESET)
      await axios.post(
        `https://api.cloudinary.com/v1_1/${CLOUD_NAME}/raw/upload`,
        cdnForm
      )

      setStage('processing')

      const backendForm = new FormData()
      backendForm.append('file', file)
      await axios.post(`${API}/upload-doc`, backendForm, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      setStage('done')
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Upload failed')
      setStage('error')
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles: 1,
    disabled: stage === 'uploading' || stage === 'processing',
  })

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`glass rounded-card-lg p-12 text-center cursor-pointer transition-all ${
          isDragActive ? 'bg-white/10 border-brand/40' : 'hover:bg-white/[0.06]'
        } ${stage === 'uploading' || stage === 'processing' ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        <p className="text-muted text-sm">
          {isDragActive
            ? 'Drop your PDF here…'
            : 'Drag & drop a PDF, or click to select'}
        </p>
        {filename && stage !== 'idle' && (
          <p className="mt-2 text-near-white text-sm font-mono">{filename}</p>
        )}
      </div>

      {STAGES[stage] && (
        <div className="flex items-center gap-2 text-sm text-muted">
          {stage !== 'done' && (
            <span className="inline-block w-3.5 h-3.5 border-2 border-white/20 border-t-brand rounded-full animate-spin" />
          )}
          <span className={stage === 'done' ? 'text-brand' : ''}>{STAGES[stage]}</span>
        </div>
      )}

      {stage === 'error' && (
        <p className="text-sm text-muted glass rounded-card px-4 py-2">{error}</p>
      )}

      {stage === 'done' && (
        <button
          onClick={() => { setStage('idle'); setFilename('') }}
          className="px-6 py-2 rounded-pill border border-white/[0.08] text-near-white text-sm font-medium hover:bg-white/5 transition-colors"
        >
          Upload another
        </button>
      )}
    </div>
  )
}
