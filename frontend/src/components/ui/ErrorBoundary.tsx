import { Component, type ErrorInfo, type ReactNode } from "react";

/** Last-resort guard: a render error in one page shows a message instead of a blank screen. */
export default class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("UI render error", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" style={{ padding: "60px 0", maxWidth: 560 }}>
        <span className="eyebrow" style={{ color: "var(--color-critical)" }}>UI_ERROR</span>
        <h1 style={{ fontSize: 24, margin: "8px 0" }}>Something went wrong displaying this page</h1>
        <p style={{ color: "var(--color-ink-muted)", fontSize: 14 }}>{this.state.error.message}</p>
        <button className="btn-primary" style={{ marginTop: 16 }} onClick={() => window.location.assign("/")}>
          Return to dashboard
        </button>
      </div>
    );
  }
}
