import { NavLink } from "react-router-dom";
import WaveformMark from "./WaveformMark";
import { useApi } from "../../hooks/useApi";
import { getHealth } from "../../api/endpoints";

const NAV_ITEMS = [
  { to: "/", label: "Overview", end: true },
  { to: "/analyze", label: "Analyze" },
  { to: "/history", label: "History" },
  { to: "/analytics", label: "Analytics" },
  { to: "/models", label: "Models" },
];

export default function TopNav() {
  const { data: health } = useApi(getHealth, []);
  const isUp = health?.status === "ok";

  return (
    <header className="top-nav">
      <div className="top-nav-inner">
        <div className="brand">
          <WaveformMark />
          <div className="brand-text">
            <span className="brand-name">AEROSENSE</span>
            <span className="brand-tagline">Acoustic Intelligence</span>
          </div>
        </div>

        <nav aria-label="Primary">
          <ul className="nav-list">
            {NAV_ITEMS.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => `nav-link${isActive ? " nav-link-active" : ""}`}
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        <div className="system-status" title={isUp ? "Backend reachable" : "Backend unreachable"}>
          <span className={`status-dot${isUp ? " status-dot-live" : ""}`} />
          <span className="eyebrow">{health === null && isUp === false ? "checking" : isUp ? "system live" : "offline"}</span>
        </div>
      </div>

      <style>{`
        .top-nav {
          position: sticky;
          top: 0;
          z-index: 20;
          backdrop-filter: blur(10px);
          background: rgba(250, 247, 241, 0.86);
          border-bottom: 1px solid var(--color-border);
        }
        .top-nav-inner {
          max-width: 1240px;
          margin: 0 auto;
          padding: 14px 28px;
          display: flex;
          align-items: center;
          gap: 32px;
        }
        .brand {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-right: 8px;
        }
        .brand-text {
          display: flex;
          flex-direction: column;
          line-height: 1.1;
        }
        .brand-name {
          font-family: var(--font-display);
          font-weight: 700;
          font-size: 15px;
          letter-spacing: 0.04em;
        }
        .brand-tagline {
          font-family: var(--font-mono);
          font-size: 10px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--color-ink-muted);
        }
        .nav-list {
          list-style: none;
          display: flex;
          gap: 4px;
          margin: 0;
          padding: 0;
          flex: 1;
        }
        .nav-link {
          display: inline-block;
          padding: 7px 12px;
          border-radius: var(--radius-sm);
          font-size: 13.5px;
          font-weight: 500;
          color: var(--color-ink-muted);
          text-decoration: none;
          transition: color 0.15s ease, background 0.15s ease;
        }
        .nav-link:hover {
          color: var(--color-ink);
          background: var(--color-canvas-sunken);
        }
        .nav-link-active {
          color: var(--color-accent-strong);
          background: var(--color-accent-soft);
        }
        .system-status {
          display: flex;
          align-items: center;
          gap: 7px;
        }
        .status-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: var(--color-ink-faint);
        }
        .status-dot-live {
          background: var(--color-success);
          box-shadow: 0 0 0 3px var(--color-success-soft);
        }
        @media (max-width: 820px) {
          .nav-list { display: none; }
        }
      `}</style>
    </header>
  );
}
