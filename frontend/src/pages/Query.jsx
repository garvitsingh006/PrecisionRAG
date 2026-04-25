import { useEffect, useState } from 'react'
import axios from 'axios'
import AnswerPanel from '../components/AnswerPanel'
import PipelineVisualizer from '../components/PipelineVisualizer'
import Loader from '../components/Loader'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function Query() {
  const [question, setQuestion] = useState('')
  const [configId, setConfigId] = useState('')
  const [configs, setConfigs] = useState([])
  const [loading, setLoading] = useState(false)
  const [pipelineStep, setPipelineStep] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    axios.get(`${API}/configs`).then(r => {
      setConfigs(r.data)
      if (r.data.length > 0) setConfigId(r.data[0].id)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!loading) { setPipelineStep(null); return }
    let step = 0
    const id = setInterval(() => {
      step = (step + 1) % 8
      setPipelineStep(step)
    }, 1800)
    return () => clearInterval(id)
  }, [loading])

  const submit = async (e) => {
    e.preventDefault()
    if (!question.trim() || !configId) return
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const { data } = await axios.post(`${API}/evaluation/new`, { question, config_id: configId })
      setResult(data)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="max-w-5xl mx-auto px-6 py-20">
      <div className="max-w-2xl space-y-8">
        <div className="space-y-2">
          <h1 className="font-display font-semibold text-4xl text-near-white tracking-[-0.8px]">Ask a Question</h1>
          <p className="text-muted text-base">Run the full RAG pipeline against your uploaded documents.</p>
        </div>

        <form onSubmit={submit} className="space-y-4">
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="Ask something about your document…"
            rows={3}
            className="w-full glass rounded-card px-4 py-3 text-sm text-near-white placeholder-muted/50 resize-none focus:outline-none focus:ring-1 focus:ring-brand/60 bg-transparent"
            required
          />
          <div className="flex items-center gap-3">
            <select
              value={configId}
              onChange={e => setConfigId(e.target.value)}
              className="glass rounded-pill px-4 py-2 text-sm text-near-white bg-transparent focus:outline-none focus:ring-1 focus:ring-brand/60"
              required
            >
              {configs.length === 0 && <option value="">No configs — create one first</option>}
              {configs.map(c => (
                <option key={c.id} value={c.id} className="bg-surface text-near-white">{c.id}</option>
              ))}
            </select>
            <button
              type="submit"
              disabled={loading || !configId}
              className="px-6 py-2 rounded-pill bg-brand text-page text-sm font-medium disabled:opacity-40 hover:opacity-90 transition-opacity"
            >
              {loading ? 'Running…' : 'Run AI'}
            </button>
          </div>
        </form>

        {loading && (
          <div className="space-y-4">
            <Loader message="Pipeline running — this may take a minute…" />
            <PipelineVisualizer active={pipelineStep} />
          </div>
        )}

        {error && (
          <p className="text-sm text-muted glass rounded-card px-4 py-3">{error}</p>
        )}

        {result && <AnswerPanel data={result} />}
      </div>
    </main>
  )
}
