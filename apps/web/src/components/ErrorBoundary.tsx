import React from "react";

/**
 * Visible error boundary for the whole app.
 *
 * Replaces the silent "dark blue screen on crash" with a readable
 * trace + a hint that hot-reload is available. Without this, an
 * unhandled render error during a child component (e.g. an inspector
 * panel, an overlay, a chart) takes the entire React tree down and
 * leaves only the body background color behind.
 *
 * The boundary deliberately uses inline styles (not Tailwind) so it
 * still renders even if a Tailwind / CSS compilation issue is the
 * thing that broke the page.
 */

interface State {
  error?: Error;
  info?: React.ErrorInfo;
}

interface Props {
  children: React.ReactNode;
}

class ErrorBoundary extends React.Component<Props, State> {
  override state: State = { error: undefined, info: undefined };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // Eslint won't let us console.error without a directive in some configs;
    // keep noisy on purpose — this is the whole point of the boundary.

    console.error("[Kivski/ErrorBoundary] render crash:", error, info);
    this.setState({ info });
  }

  override render(): React.ReactNode {
    const { error, info } = this.state;
    if (!error) return this.props.children;

    return (
      <div
        style={{
          padding: 24,
          color: "#ff7a7a",
          fontFamily:
            "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
          background: "#0a0e14",
          minHeight: "100vh",
          overflow: "auto",
        }}
      >
        <h1 style={{ color: "#FFC833", margin: 0, fontSize: 18 }}>
          Kivski Frontend Crash
        </h1>
        <p style={{ opacity: 0.75, marginTop: 4, fontSize: 12 }}>
          A React render threw and the app would otherwise show a blank screen.
        </p>

        <h2 style={{ color: "#FFC833", marginTop: 18, fontSize: 14 }}>
          Error
        </h2>
        <pre
          style={{
            margin: 0,
            padding: 12,
            background: "#131821",
            border: "1px solid #222B3A",
            borderRadius: 4,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {error.name}: {error.message}
        </pre>

        {error.stack && (
          <>
            <h2 style={{ color: "#FFC833", marginTop: 18, fontSize: 14 }}>
              Stack
            </h2>
            <pre
              style={{
                margin: 0,
                padding: 12,
                background: "#131821",
                border: "1px solid #222B3A",
                borderRadius: 4,
                fontSize: 11,
                opacity: 0.9,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {error.stack}
            </pre>
          </>
        )}

        {info?.componentStack && (
          <>
            <h2 style={{ color: "#FFC833", marginTop: 18, fontSize: 14 }}>
              Component stack
            </h2>
            <pre
              style={{
                margin: 0,
                padding: 12,
                background: "#131821",
                border: "1px solid #222B3A",
                borderRadius: 4,
                fontSize: 11,
                opacity: 0.85,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {info.componentStack}
            </pre>
          </>
        )}

        <p style={{ opacity: 0.7, marginTop: 18, fontSize: 12 }}>
          Open the browser devtools console for the full trace. Saving any
          source file triggers a hot-reload — no manual reload needed.
        </p>
      </div>
    );
  }
}

export default ErrorBoundary;
