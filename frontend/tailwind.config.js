/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        body: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['Geist Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        page: '#0d0d0d',
        surface: '#141414',
        'surface-2': '#1a1a1a',
        'near-white': '#ededed',
        muted: '#a0a0a0',
        subtle: '#666666',
        brand: '#18E299',
        'brand-deep': '#0fa76e',
        'brand-light': '#d4fae8',
        'border-dim': 'rgba(255,255,255,0.08)',
        'border-subtle': 'rgba(255,255,255,0.05)',
      },
      borderRadius: {
        card: '16px',
        'card-lg': '24px',
        pill: '9999px',
      },
      boxShadow: {
        glass: '0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)',
        card: '0 2px 4px rgba(0,0,0,0.4)',
      },
      backdropBlur: {
        glass: '16px',
      },
    },
  },
  plugins: [],
}
