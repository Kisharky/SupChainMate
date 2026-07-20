/**
 * SupChainMate — Tailwind theme extension
 * Merge into tailwind.config.{js,ts} `theme.extend`. Every value maps to a
 * CSS variable in design/tokens.css so utilities stay theme-aware.
 *
 *   const tokens = require("./design/tailwind.tokens");
 *   module.exports = { theme: { extend: tokens } };
 */
module.exports = {
  colors: {
    navy: { 950:"#060B16",900:"#0B1220",850:"#0E1729",800:"#111A2E",700:"#16213A",600:"#1E2A44",500:"#2A3A5C" },
    slate: { 50:"#F5F7FA",100:"#E9EDF3",200:"#D4DBE6",300:"#B4BECE",400:"#8A94A6",500:"#66707F",600:"#4B5563",700:"#333B49" },
    emerald: { 400:"#34D399",500:"#10B981",600:"#059669" },
    good:"#10B981", warning:"#F59E0B", critical:"#EF4444", info:"#38BDF8",
    // Semantic aliases bound to the theme variables
    bg:"var(--bg)", "bg-sunken":"var(--bg-sunken)", panel:"var(--panel)", "panel-2":"var(--panel-2)",
    rail:"var(--rail)", hairline:"var(--hairline)", "hairline-strong":"var(--hairline-strong)",
    ink:"var(--text)", "ink-2":"var(--text-2)", "ink-3":"var(--text-3)",
    accent:"var(--accent)", "accent-ink":"var(--accent-ink)", focus:"var(--focus)",
  },
  fontFamily: {
    sans: ["Inter","-apple-system","BlinkMacSystemFont","Segoe UI","Roboto","sans-serif"],
    mono: ["SFMono-Regular","ui-monospace","JetBrains Mono","Menlo","Consolas","monospace"],
  },
  fontSize: {
    "2xs":".75rem", xs:".8125rem", sm:".9375rem", base:"1.125rem",
    lg:"1.5rem", xl:"2.25rem", "2xl":"3rem",
  },
  borderRadius: { sm:"5px", DEFAULT:"8px", lg:"12px" },
  boxShadow: { card:"var(--shadow)" },
};
