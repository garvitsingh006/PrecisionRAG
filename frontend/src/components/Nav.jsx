import { Link, useLocation } from 'react-router-dom'

const links = [
  { to: '/upload', label: 'Upload' },
  { to: '/query', label: 'Query' },
  { to: '/config', label: 'Config' },
  { to: '/resume', label: 'Resume' },
]

export default function Nav() {
  const { pathname } = useLocation()

  return (
    <nav className="sticky top-0 z-50 border-b border-white/[0.08] bg-[#0d0d0d]/80 backdrop-blur-glass">
      <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
        <Link to="/" className="font-display font-semibold text-near-white text-base tracking-tight">
          Precision <span className="text-brand">RAG</span>
        </Link>
        <div className="flex items-center gap-1">
          {links.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              className={`px-4 py-1.5 rounded-pill text-sm font-medium transition-colors ${
                pathname === to
                  ? 'bg-white/10 text-near-white'
                  : 'text-muted hover:text-near-white'
              }`}
            >
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}
