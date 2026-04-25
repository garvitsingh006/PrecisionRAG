import { useState } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const defaults = {
  id: '',
  model: 'deepseek-ai/DeepSeek-V3',
  embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
  chunk_size: 700,
  chunk_overlap: 150,
  top_k: 4,
  temperature: 0.7,
}

export default function ConfigForm({ onCreated }) {
  const [form, setForm] = useState(defaults)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const { data } = await axios.post(`${API}/config/new`, {
        ...form,
        chunk_size: Number(form.chunk_size),
        chunk_overlap: Number(form.chunk_overlap),
        top_k: Number(form.top_k),
        temperature: Number(form.temperature),
      })
      onCreated?.(data)
      setForm(defaults)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const field = (label, key, type = 'text', step) => (
    <div className="space-y-1">
      <label className="text-xs font-mono text-brand/60 uppercase tracking-widest">{label}</label>
      <input
        type={type}
        step={step}
        value={form[key]}
        onChange={e => set(key, e.target.value)}
        className="w-full glass rounded-pill px-4 py-2 text-sm text-near-white bg-transparent placeholder-muted/50 focus:outline-none focus:ring-1 focus:ring-brand/60"
        required
      />
    </div>
  )

  return (
    <form onSubmit={submit} className="space-y-4">
      {field('Config ID', 'id')}
      {field('Model', 'model')}
      {field('Embedding Model', 'embedding_model')}
      <div className="grid grid-cols-2 gap-4">
        {field('Chunk Size', 'chunk_size', 'number')}
        {field('Chunk Overlap', 'chunk_overlap', 'number')}
        {field('Top K', 'top_k', 'number')}
        {field('Temperature', 'temperature', 'number', '0.1')}
      </div>
      {error && <p className="text-sm text-muted glass rounded-card px-4 py-2">{error}</p>}
      <button
        type="submit"
        disabled={loading}
        className="px-6 py-2 rounded-pill bg-brand text-page text-sm font-medium disabled:opacity-40 hover:opacity-90 transition-opacity"
      >
        {loading ? 'Creating…' : 'Create Config'}
      </button>
    </form>
  )
}
