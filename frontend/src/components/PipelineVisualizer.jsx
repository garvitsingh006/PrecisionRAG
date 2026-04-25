const STEPS = [
  'Decide Retrieval',
  'Rewrite Query',
  'Retrieve Docs',
  'Evaluate Docs',
  'Refine Context',
  'Generate Answer',
  'Check Hallucination',
  'Check Usefulness',
]

export default function PipelineVisualizer({ active }) {
  return (
    <div className="glass rounded-card p-5">
      <p className="text-xs font-mono text-brand/60 uppercase tracking-widest mb-4">Pipeline</p>
      <div className="flex flex-wrap gap-2 items-center">
        {STEPS.map((step, i) => (
          <div key={step} className="flex items-center gap-2">
            <span
              className={`px-3 py-1 rounded-pill text-xs font-mono transition-all ${
                active != null
                  ? i === active
                    ? 'bg-brand text-page'
                    : i < active
                    ? 'bg-brand/20 text-brand/70'
                    : 'bg-white/5 border border-white/[0.08] text-muted'
                  : 'bg-white/5 border border-white/[0.08] text-muted'
              }`}
            >
              {step}
            </span>
            {i < STEPS.length - 1 && <span className="text-muted/40 text-xs">→</span>}
          </div>
        ))}
      </div>
    </div>
  )
}
