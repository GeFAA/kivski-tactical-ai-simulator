import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import ErrorBoundary from "./components/ErrorBoundary";
import "./index.css";

// ---------- Global runtime diagnostics ----------
//
// Without these, any uncaught error or unhandled promise rejection from
// (e.g.) a Pixi async init, a WebSocket reconnect, or a tab-hidden
// store callback silently disappears into the console — and the app
// looks like it "just went dark blue".
//
// Keep these registered before render() so we also capture errors that
// fire during the very first mount.

window.addEventListener("error", (e) => {

  console.error(
    "[kivski/window.error]",
    e.error?.message ?? e.message,
    e.error?.stack,
  );
});

window.addEventListener("unhandledrejection", (e) => {

  console.error("[kivski/unhandledrejection]", e.reason);
});

// ---------- Mount ----------

const rootEl = document.getElementById("root");
if (!rootEl) {
  throw new Error(
    "Kivski: #root mount node missing in index.html — refusing to start.",
  );
}

ReactDOM.createRoot(rootEl).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
