import type { FaultType, PredictionStatus, Severity } from "../api/types";

export function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatRelativeTime(iso: string): string {
  const d = new Date(iso).getTime();
  const diffSec = Math.round((Date.now() - d) / 1000);
  if (diffSec < 60) return "just now";
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return `${Math.floor(diffSec / 86400)}d ago`;
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatScore(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined) return "—";
  return value.toFixed(digits);
}

export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export function formatFaultLabel(fault: FaultType | null): string {
  if (!fault) return "—";
  const map: Record<string, string> = {
    "1_broken": "1 blade damaged",
    "2_broken": "2 blades damaged",
    "3_4_broken": "3–4 blades damaged",
  };
  return map[fault] ?? fault;
}

export function formatClassLabel(className: string): string {
  const map: Record<string, string> = {
    normal: "Normal",
    "1_broken": "1 blade damaged",
    "2_broken": "2 blades damaged",
    "3_4_broken": "3–4 blades damaged",
  };
  return map[className] ?? className;
}

// Design tokens for status/severity, kept in one place so color logic
// never drifts between components.
export const STATUS_COLOR: Record<PredictionStatus, string> = {
  NORMAL: "var(--color-success)",
  ANOMALY: "var(--color-critical)",
  UNKNOWN_ANOMALY: "var(--color-warning)",
  ERROR: "var(--color-critical)",
};

export const SEVERITY_COLOR: Record<Severity, string> = {
  LOW: "var(--color-success)",
  MEDIUM: "var(--color-warning)",
  HIGH: "var(--color-critical)",
  CRITICAL: "var(--color-critical)",
};

export const STATUS_LABEL: Record<PredictionStatus, string> = {
  NORMAL: "Normal",
  ANOMALY: "Anomaly detected",
  UNKNOWN_ANOMALY: "Unknown anomaly",
  ERROR: "Error",
};
