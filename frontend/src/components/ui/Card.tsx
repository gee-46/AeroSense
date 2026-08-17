import type { CSSProperties, ReactNode } from "react";

export default function Card({
  children,
  style,
  padded = true,
  className = "",
}: {
  children: ReactNode;
  style?: CSSProperties;
  padded?: boolean;
  className?: string;
}) {
  return (
    <div
      className={`aero-card ${className}`}
      style={{
        background: "var(--color-canvas-raised)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-soft)",
        padding: padded ? "22px 24px" : 0,
        ...style,
      }}
    >
      {children}
    </div>
  );
}
