import type { ApiErrorBody } from "./types";

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ??
  "http://localhost:8000";

export class ApiError extends Error {
  code: string;
  status: number;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function parseErrorBody(res: Response): Promise<{ code: string; message: string }> {
  try {
    const body = (await res.json()) as ApiErrorBody;
    if (body?.error?.code && body?.error?.message) {
      return body.error;
    }
  } catch {
    // response wasn't JSON (e.g. a network-level failure) — fall through
  }
  return { code: "UNKNOWN_ERROR", message: `Request failed with status ${res.status}` };
}

/** fetch() rejects (TypeError) when the server is unreachable or CORS blocks
 * the call; surface that as a readable ApiError instead of "Failed to fetch". */
async function request(url: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(0, "NETWORK_ERROR", `Cannot reach the AeroSense API at ${API_BASE_URL}. Is the backend running?`);
  }
}

export async function apiGet<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(`${API_BASE_URL}${path}`);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== "") url.searchParams.set(key, String(value));
    });
  }
  const res = await request(url.toString());
  if (!res.ok) {
    const { code, message } = await parseErrorBody(res);
    throw new ApiError(res.status, code, message);
  }
  return res.json() as Promise<T>;
}

export async function apiPostForm<T>(path: string, formData: FormData): Promise<T> {
  const res = await request(`${API_BASE_URL}${path}`, { method: "POST", body: formData });
  if (!res.ok) {
    const { code, message } = await parseErrorBody(res);
    throw new ApiError(res.status, code, message);
  }
  return res.json() as Promise<T>;
}

export async function apiDelete(path: string): Promise<void> {
  const res = await request(`${API_BASE_URL}${path}`, { method: "DELETE" });
  if (!res.ok && res.status !== 204) {
    const { code, message } = await parseErrorBody(res);
    throw new ApiError(res.status, code, message);
  }
}

/** Resolves a server-relative URL path (e.g. Grad-CAM heatmap_path) against
 * the configured API base URL, so <img src> works regardless of deployment. */
export function resolveApiUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE_URL}${path}`;
}

/** Mirrors the backend's MAX_UPLOAD_SIZE_MB default; the server remains the authority. */
export const MAX_UPLOAD_MB = 25;
