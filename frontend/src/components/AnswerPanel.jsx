function ScoreBar({ value }) {
  const pct = Math.round((value ?? 0) * 100)
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-1.5 bg-white/10 rounded-pill overflow-hidden">
        <div className="h-full bg-brand rounded-pill transition-all" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-muted w-8 text-right">{pct}%</span>
    </div>
  )
}

function Tag({ label }) {
  const map = {
    fully_supported: 'Fully Supported',
    partially_supported: 'Partially Supported',
    no_support: 'No Support',
    useful: 'Useful',
    not_useful: 'Not Useful',
  }
  return (
    <span className="px-3 py-0.5 rounded-pill bg-brand/10 text-brand text-xs font-mono">
      {map[label] ?? label}
    </span>
  )
}

function Section({ title, children }) {
  return (
    <div className="glass rounded-card p-5 space-y-3">
      <p className="text-xs font-mono text-brand/60 uppercase tracking-widest">{title}</p>
      {children}
    </div>
  )
}

export default function AnswerPanel({ data }) {
  const { answer, evaluation, pipeline, performance, experiment } = data

  return (
    <div className="space-y-4">
      <Section title="Answer">
        <p className="text-near-white text-base leading-relaxed">{answer}</p>
      </Section>

      {evaluation && (
        <Section title="Evaluation">
          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-muted">Confidence</span>
              </div>
              <ScoreBar value={evaluation.confidence} />
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-muted">Retrieval Relevance</span>
              </div>
              <ScoreBar value={evaluation.retrieval_relevance} />
            </div>
            {evaluation.support && (
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted">Support</span>
                  <Tag label={evaluation.support.label} />
                </div>
                <ScoreBar value={evaluation.support.score} />
                {evaluation.support.reason && (
                  <p className="text-xs text-subtle">{evaluation.support.reason}</p>
                )}
              </div>
            )}
            {evaluation.usefulness && (
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted">Usefulness</span>
                  <Tag label={evaluation.usefulness.label} />
                </div>
                <ScoreBar value={evaluation.usefulness.score} />
                {evaluation.usefulness.reason && (
                  <p className="text-xs text-subtle">{evaluation.usefulness.reason}</p>
                )}
              </div>
            )}
          </div>
        </Section>
      )}

      <div className="grid grid-cols-2 gap-4">
        {pipeline && (
          <Section title="Pipeline">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted">Retrieval used</span>
                <span className={`font-mono ${pipeline.retrieval_used ? 'text-brand' : 'text-subtle'}`}>{pipeline.retrieval_used ? '✓' : '✗'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Web search used</span>
                <span className={`font-mono ${pipeline.web_search_used ? 'text-brand' : 'text-subtle'}`}>{pipeline.web_search_used ? '✓' : '✗'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Hallucination retries</span>
                <span className="font-mono text-near-white">{pipeline.hallucination_retries}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Usefulness retries</span>
                <span className="font-mono text-near-white">{pipeline.usefulness_retries}</span>
              </div>
            </div>
          </Section>
        )}
        {performance && (
          <Section title="Performance">
            <div className="flex justify-between text-sm">
              <span className="text-muted">Latency</span>
              <span className="font-mono text-near-white">{performance.latency_ms} ms</span>
            </div>
          </Section>
        )}
      </div>

      {experiment && (
        <Section title="Experiment Config">
          <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm">
            {Object.entries(experiment).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-muted">{k}</span>
                <span className="font-mono text-near-white truncate max-w-[160px]" title={v}>{v}</span>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  )
}
