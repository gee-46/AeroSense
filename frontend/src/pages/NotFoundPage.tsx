import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <div style={{ padding: "60px 0", maxWidth: 520 }}>
      <span className="eyebrow">404</span>
      <h1 style={{ fontSize: 28, margin: "8px 0" }}>Page not found</h1>
      <p style={{ color: "var(--color-ink-muted)", fontSize: 14, marginBottom: 16 }}>
        There is nothing at this address.
      </p>
      <Link to="/" className="btn-primary">
        Go to dashboard
      </Link>
    </div>
  );
}
