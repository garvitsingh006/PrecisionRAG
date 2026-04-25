export default function Loader({ message = 'Loading…' }) {
  return (
    <div className="flex items-center gap-2 text-muted text-sm">
      <span className="inline-block w-4 h-4 border-2 border-white/20 border-t-brand rounded-full animate-spin" />
      {message}
    </div>
  )
}
