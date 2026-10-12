import { apiGet, apiPostForm, apiDelete } from "./client";
import type {
  HealthResponse,
  ModelHealthResponse,
  AudioMetadata,
  PredictionResult,
  PredictionListResponse,
  ModelsListResponse,
  ActiveModelsResponse,
  AnalyticsSummary,
  FaultDistribution,
  AnomalyTrendPoint,
  AnalyzeParams,
} from "./types";

// All paths below were read from the live backend's /openapi.json — none
// are invented. Every route is prefixed /api/v1, which differs from the
// unprefixed paths in earlier planning notes; the prefixed versions are
// what the running FastAPI app actually serves.

export const getHealth = () => apiGet<HealthResponse>("/api/v1/health");

export const getModelHealth = () => apiGet<ModelHealthResponse>("/api/v1/health/model");

export async function uploadAudio(file: File): Promise<AudioMetadata> {
  const form = new FormData();
  form.append("file", file);
  return apiPostForm<AudioMetadata>("/api/v1/audio/upload", form);
}

export async function analyzeAudio(file: File, params: AnalyzeParams): Promise<PredictionResult> {
  const form = new FormData();
  form.append("file", file);
  if (params.device_id) form.append("device_id", params.device_id);
  if (params.environment) form.append("environment", params.environment);
  if (params.operating_condition) form.append("operating_condition", params.operating_condition);
  if (params.rpm !== undefined) form.append("rpm", String(params.rpm));
  return apiPostForm<PredictionResult>("/api/v1/audio/analyze", form);
}

export function listPredictions(opts: {
  limit?: number;
  offset?: number;
  status?: string;
  device_id?: string;
}): Promise<PredictionListResponse> {
  return apiGet<PredictionListResponse>("/api/v1/predictions", opts);
}

export const getPrediction = (id: string) => apiGet<PredictionResult>(`/api/v1/predictions/${encodeURIComponent(id)}`);

export const deletePrediction = (id: string) => apiDelete(`/api/v1/predictions/${encodeURIComponent(id)}`);

export const listModels = () => apiGet<ModelsListResponse>("/api/v1/models");

export const getActiveModels = () => apiGet<ActiveModelsResponse>("/api/v1/models/active");

export const getAnalyticsSummary = () => apiGet<AnalyticsSummary>("/api/v1/analytics/summary");

export const getFaultDistribution = () => apiGet<FaultDistribution>("/api/v1/analytics/fault-distribution");

export const getAnomalyTrend = () => apiGet<AnomalyTrendPoint[]>("/api/v1/analytics/anomaly-trend");
