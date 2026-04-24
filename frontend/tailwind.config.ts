import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // LucidCredit brand — deep navy + electric teal
        brand: {
          50: "#edfafa",
          100: "#d5f5f6",
          200: "#afeaed",
          300: "#76d8de",
          400: "#3bbfc9",
          500: "#1ea2af", // primary
          600: "#1a8794",
          700: "#1b6d78",
          800: "#1d5863",
          900: "#1b4a53",
          950: "#0c3039",
        },
        // Decision semantic
        approve: { DEFAULT: "#10b981", muted: "#064e3b", text: "#34d399" },
        decline: { DEFAULT: "#ef4444", muted: "#7f1d1d", text: "#fca5a5" },
        pending: { DEFAULT: "#f59e0b", muted: "#78350f", text: "#fcd34d" },
        // Confidence bands
        confidence: {
          high: "#10b981",   // >= 0.90
          medium: "#f59e0b", // 0.75 – 0.89
          low: "#ef4444",    // < 0.75
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Menlo", "monospace"],
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "fade-in": {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "fade-in": "fade-in 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
