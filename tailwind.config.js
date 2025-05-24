/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./*.html",          // Adjust this to match your actual file paths
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#e0f2fe',
          100: '#bae6fd',
          200: '#7dd3fc',
          300: '#38bdf8',
          400: '#0ea5e9',
          500: '#0284c7', // Main primary color
          600: '#0369a1',
          700: '#075985',
          800: '#0c4a6e',
          900: '#0a3446',
        },
        secondary: {
          50: '#f1f5f9',
          100: '#e2e8f0',
          200: '#cbd5e1',
          300: '#94a3b8',
          400: '#64748b',
          500: '#475569', // Main secondary color
          600: '#334155',
          700: '#1e293b',
          800: '#0f172a',
          900: '#0a0e1a',
        },
        accent: {
          50: '#fff7ed',
          100: '#ffedd5',
          200: '#fed7aa',
          300: '#fdba74',
          400: '#fb923c',
          500: '#f59e0b', // Main accent color
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
          900: '#78350f',
        },
      },
    },
  },
  plugins: [],
}