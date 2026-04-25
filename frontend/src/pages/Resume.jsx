import { useState } from 'react'
import axios from 'axios'
import AnswerPanel from '../components/AnswerPanel'
import Loader from '../components/Loader'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function Resume() {
  const [runId, setRunId] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    if (!runId) return
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const { data } = await axios.post(`${API}/evaluation/resume/${runId}`)
      setResult(data)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Resume failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="max-w-5xl mx-auto px-6 py-20">
      <div className="max-w-2xl space-y-8">
        <div className="space-y-2">
          <h1 className="font-display font-semibold text-4xl text-near-white tracking-[-0.8px]">Resume Run</h1>
          <p className="text-muted text-base">
            Resume a <span className="font-mono text-brand">failed</span> or{' '}
            <span className="font-mono text-brand">pending</span> run from its last
            PostgreSQL checkpoint — no tokens wasted re-running completed nodes.
          </p>
        </div>

        <form onSubmit={submit} className="flex items-center gap-3">
          <input
            type="number"
            value={runId}
            onChange={e => setRunId(e.target.value)}
            placeholder="Run ID"
            className="glass rounded-pill px-4 py-2 text-sm text-near-white bg-transparent placeholder-muted/50 w-36 focus:outline-none focus:ring-1 focus:ring-brand/60"
            required
          />
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2 rounded-pill bg-brand text-page text-sm font-medium disabled:opacity-40 hover:opacity-90 transition-opacity"
          >
            {loading ? 'Resuming…' : 'Resume'}
          </button>
        </form>

        {loading && <Loader message="Resuming from checkpoint…" />}

        {error && (
          <p className="text-sm text-muted glass rounded-card px-4 py-3">{error}</p>
        )}

        {result && <AnswerPanel data={result} />}
      </div>
    </main>
  )
}
