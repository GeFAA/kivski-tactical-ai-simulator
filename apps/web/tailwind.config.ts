import type { Config } from "tailwindcss";

/**
 * Tailwind theme for the Kivski Tactical AI Simulator.
 *
 * Color palette references the in-game team colors:
 *  - attackers: yellow #FFC833
 *  - defenders: blue   #4DA8FF
 * The surface tones are a dark, low-contrast game-HUD style so the
 * map and player dots stay visually dominant.
 */
const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        kivski: {
          bg: "#0A0E14",
          panel: "#131821",
          "panel-2": "#1A2030",
          border: "#222B3A",
          muted: "#6B7585",
          text: "#E5E7EB",
          // Teams
          attacker: "#FFC833",
          "attacker-dim": "#A8821C",
          defender: "#4DA8FF",
          "defender-dim": "#2F6FB0",
          // Status
          hp: "#4ADE80",
          "hp-low": "#F87171",
          armor: "#60A5FA",
          money: "#FACC15",
          // Map
          wall: "#2A2F3A",
          cover: "#3B4252",
          siteA: "#FF6B6B",
          siteB: "#6BCB77",
          bomb: "#FF8C42",
        },
      },
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
      },
      boxShadow: {
        panel: "inset 0 0 0 1px rgba(255,255,255,0.04)",
        glow: "0 0 12px rgba(77,168,255,0.35)",
      },
      animation: {
        "pulse-slow": "pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
    },
  },
  plugins: [],
};

export default config;
