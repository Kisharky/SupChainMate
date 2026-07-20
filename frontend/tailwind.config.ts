import type { Config } from "tailwindcss";

/**
 * Theme bound to the design tokens in app/globals.css. Colors reference CSS
 * variables so every utility is theme-aware (light/dark) automatically.
 */
const config: Config = {
  darkMode: ["class", '[data-theme="dark"]'],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: { 950: "#060B16", 900: "#0B1220", 850: "#0E1729", 800: "#111A2E", 700: "#16213A", 600: "#1E2A44", 500: "#2A3A5C" },
        slate: { 50: "#F5F7FA", 100: "#E9EDF3", 200: "#D4DBE6", 300: "#B4BECE", 400: "#8A94A6", 500: "#66707F", 600: "#4B5563", 700: "#333B49" },
        emerald: { 400: "#34D399", 500: "#10B981", 600: "#059669" },
        good: "#10B981", warning: "#F59E0B", critical: "#EF4444", info: "#38BDF8",
        bg: "var(--bg)", sunken: "var(--bg-sunken)", panel: "var(--panel)", "panel-2": "var(--panel-2)",
        rail: "var(--rail)", hairline: "var(--hairline)", "hairline-strong": "var(--hairline-strong)",
        ink: "var(--text)", "ink-2": "var(--text-2)", "ink-3": "var(--text-3)",
        accent: "var(--accent)", "accent-ink": "var(--accent-ink)", focus: "var(--focus)",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "Inter", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "JetBrains Mono", "ui-monospace", "monospace"],
      },
      borderRadius: { sm: "5px", DEFAULT: "8px", lg: "12px" },
      boxShadow: { card: "var(--shadow)" },
    },
  },
  plugins: [],
};
export default config;
