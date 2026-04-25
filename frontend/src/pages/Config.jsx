import { useEffect, useState } from 'react'
import axios from 'axios'
import ConfigForm from '../components/ConfigForm'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function Config() {
  const [configs, setConfigs] = useState([])

  const load = () => axios.get(`${API}/configs`).then(r => setConfigs(r.data)).catch(() => {})

  useEffect(() => { load() }, [])

  return (
    <main className="max-w-5xl mx-auto px-6 py-20">
      <div className="max-w-2xl space-y-12">
        <div className="space-y-2">
          <h1 className="font-display font-semibold text-4xl text-near-white tracking-[-0.8px]">Experiment Configs</h1>
          <p className="text-muted text-base">
            Each config controls the model, chunk size, top-k, and temperature for a run.
          </p>
        </div>

        <ConfigForm onCreated={(c) => setConfigs(prev => [...prev, c])} />

        {configs.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs font-mono text-brand/60 uppercase tracking-widest">Existing Configs</p>
            {configs.map(c => (
              <div key={c.id} className="glass rounded-card p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-near-white font-mono">{c.id}</span>
                  <span className="text-xs text-muted font-mono">temp {c.temperature}</span>
                </div>
                <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-xs text-muted font-mono">
                  <span>model: <span className="text-near-white">{c.model}</span></span>
                  <span>top_k: <span className="text-near-white">{c.top_k}</span></span>
                  <span>chunk_size: <span className="text-near-white">{c.chunk_size}</span></span>
                  <span>chunk_overlap: <span className="text-near-white">{c.chunk_overlap}</span></span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  )
}
