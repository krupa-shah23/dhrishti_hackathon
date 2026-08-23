/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        // DRISHTI teal brand palette — mirrors CSS variables
        teal: {
          50:  '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6',  // --accent-blue / primary
          600: '#0d9488',  // --status-blue
          700: '#0f766e',  // --navy2
          800: '#115e59',
          900: '#134e4a',  // sidebar background
          950: '#042f2e',  // --navy (deepest)
        },
        brand: {
          bg:      '#f8f9fb',   // --bg
          card:    '#ffffff',   // --card
          border:  '#e1e2e4',   // --border
          text:    '#191c1e',   // --text
          muted:   '#676975',   // --text-muted
          sub:     '#45464e',   // --text-secondary
        },
      },
      fontFamily: {
        sans: ['Inter', 'Arial', 'sans-serif'],
      },
      borderRadius: {
        'sm':   '8px',
        'md':   '12px',
        'lg':   '16px',
        'xl':   '20px',
        'full': '999px',
      },
      boxShadow: {
        'card':       '0 4px 20px rgba(30,42,74,.05)',
        'teal-glow':  '0 0 40px rgba(20,184,166,.18)',
        'teal-soft':  '0 0 80px rgba(20,184,166,.10)',
        'teal-hover': '0 8px 30px rgba(20,184,166,.25)',
      },
      backgroundImage: {
        'teal-gradient':        'linear-gradient(135deg, #14b8a6 0%, #0f766e 100%)',
        'teal-gradient-light':  'linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)',
        'hero-radial':          'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(20,184,166,0.15), transparent)',
        'card-glow':            'radial-gradient(circle at top right, rgba(20,184,166,0.08), transparent 60%)',
      },
      animation: {
        'slide-up':    'slideUp 0.4s ease both',
        'fade-in':     'fadeIn 0.5s ease both',
        'pulse-slow':  'pulse 3s ease-in-out infinite',
        'glow-pulse':  'glowPulse 2s ease-in-out infinite',
        'float':       'float 6s ease-in-out infinite',
      },
      keyframes: {
        slideUp: {
          from: { opacity: '0', transform: 'translateY(20px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(20,184,166,0.1)' },
          '50%':      { boxShadow: '0 0 40px rgba(20,184,166,0.25)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':      { transform: 'translateY(-8px)' },
        },
      },
    },
  },
  plugins: [],
};
