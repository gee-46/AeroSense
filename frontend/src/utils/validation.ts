export const MAX_RPM = 200000;

/** Returns an error message for an invalid RPM, or null. Mirrors the backend (0..200000). */
export function validateRpm(rpm: number | undefined): string | null {
  if (rpm === undefined) return null;
  if (!Number.isFinite(rpm) || rpm < 0 || rpm > MAX_RPM) {
    return `RPM must be between 0 and ${MAX_RPM.toLocaleString()}.`;
  }
  return null;
}
