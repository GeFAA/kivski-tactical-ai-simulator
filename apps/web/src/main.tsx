import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

const rootEl = document.getElementById("root");
if (!rootEl) {
  throw new Error(
    "Kivski: #root mount node missing in index.html — refusing to start.",
  );
}

ReactDOM.createRoot(rootEl).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
