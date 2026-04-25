import { Link } from 'react-router-dom'

const PIPELINE_STEPS = [
  { label: 'Decide', desc: 'Skip retrieval for timeless questions' },
  { label: 'Rewrite', desc: 'Optimize query for vector search' },
  { label: 'Retrieve', desc: 'Fetch top-k chunks from FAISS' },
  { label: 'Evaluate', desc: 'Score each chunk 0–1 for relevance' },
  { label: 'Web Search', desc: 'Fallback to Tavily if docs score low' },
  { label: 'Refine', desc: 'Strip-filter context chunk by chunk' },
  { label: 'Generate', desc: 'Answer grounded in refined context' },
  { label: 'Verify', desc: 'Hallucination check + revision loop' },
]

export default function Landing() {
  return (
    <main className="max-w-5xl mx-auto px-6 py-24 space-y-24">

      {/* Hero */}
      <section className="relative text-center space-y-6">
        <div className="hero-glow" />
        <div className="relative">
          <span className="inline-block px-3 py-1 rounded-pill bg-brand/10 text-brand text-xs font-mono uppercase tracking-widest mb-4">
            LangGraph · CRAG · Self-RAG
          </span>
          <h1 className="font-display font-semibold text-5xl text-near-white leading-tight tracking-[-1.28px]">
            Precision RAG
          </h1>
          <p className="text-muted text-lg max-w-xl mx-auto leading-relaxed mt-4">
            Hallucination-resistant question answering over your documents —
            built with LangGraph, CRAG, and Self-RAG.
          </p>
          <div className="flex justify-center gap-3 mt-6">
            <Link
              to="/query"
              className="px-6 py-2.5 rounded-pill bg-brand text-page text-sm font-medium hover:opacity-90 transition-opacity"
            >
              Try Demo
            </Link>
            <Link
              to="/upload"
              className="px-6 py-2.5 rounded-pill border border-white/[0.08] text-near-white text-sm font-medium hover:bg-white/5 transition-colors"
            >
              Upload Document
            </Link>
          </div>
        </div>
      </section>

      {/* Pipeline diagram */}
      <section className="space-y-6" id="pipeline">
        <h2 className="font-display font-semibold text-3xl text-near-white tracking-[-0.8px]">How it works</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {PIPELINE_STEPS.map((s, i) => (
            <div key={s.label} className="glass rounded-card p-4 space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-brand/60">{String(i + 1).padStart(2, '0')}</span>
                <span className="text-sm font-medium text-near-white">{s.label}</span>
              </div>
              <p className="text-xs text-muted leading-snug">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Feature highlights */}
      <section className="space-y-6">
        <h2 className="font-display font-semibold text-3xl text-near-white tracking-[-0.8px]">What makes it different</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { title: 'Hallucination Detection', body: 'Every claim in the answer is verified against the retrieved context. Unsupported answers are revised up to 3 times.' },
            { title: 'Confidence Scoring', body: 'A composite score across support (50%), usefulness (30%), and retrieval relevance (20%) tells you exactly how much to trust the answer.' },
            { title: 'Checkpoint Resume', body: 'Failed or interrupted runs are persisted in PostgreSQL and can be resumed from their last node — no tokens wasted.' },
          ].map(f => (
            <div key={f.title} className="glass-strong rounded-card-lg p-6 space-y-2">
              <p className="text-sm font-semibold text-near-white tracking-[-0.2px]">{f.title}</p>
              <p className="text-sm text-muted leading-relaxed">{f.body}</p>
            </div>
          ))}
        </div>
      </section>

    </main>
  )
}
