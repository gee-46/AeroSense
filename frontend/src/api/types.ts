// Types mirror backend/app/schemas/*.py exactly (verified against the live
// OpenAPI schema at /openapi.json). Do not add fields the API doesn't return.

export type PredictionStatus = "NORMAL" | "ANOMALY" | "UNKNOWN_ANOMALY" | "ERROR";
export type Severity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type FaultType = "1_broken" | "2_broken" | "3_4_broken" | string;

export interface AudioMetadata {
  filename: string;
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  format: string;
  size_bytes: number;
}

export interface Explanation {
  method: string;
  important_frequency_range_hz: [number, number] | null;
  important_time_range_seconds: [number, number] | null;
  heatmap_path: string | null; // URL path, e.g. "/static/outputs/gradcam_<id>.png"
  notes: string | null;
}

export interface PredictionResult {
  prediction_id: string;
  status: PredictionStatus;
  fault_type: FaultType | null;
  fault_confidence: number | null;
  anomaly_score: number;
  anomaly_threshold: number;
  severity: Severity | null;
  model_version: string;
  autoencoder_version: string;
  processing_time_ms: number;
  timestamp: string;
  audio: AudioMetadata;
  class_probabilities: Record<string, number>;
  explanation: Explanation | null;
  device_id: string | null;
  environment: string | null;
  operating_condition: string | null;
  rpm: number | null;
}

export interface PredictionListItem {
  prediction_id: string;
  status: PredictionStatus;
  fault_type: FaultType | null;
  fault_confidence: number | null;
  anomaly_score: number;
  severity: Severity | null;
  model_version: string;
  timestamp: string;
  device_id: string | null;
}

export interface PredictionListResponse {
  total: number;
  items: PredictionListItem[];
}

export interface HealthResponse {
  status: string;
  app_env: string;
  version: string;
}

export interface ModelHealthResponse {
  classifier_loaded: boolean;
  classifier_version: string | null;
  classifier_is_trained: boolean;
  autoencoder_loaded: boolean;
  autoencoder_version: string | null;
  autoencoder_is_trained: boolean;
  device: string;
  anomaly_threshold: number | null;
  details: unknown;
}

export interface ModelInfo {
  name: string;
  type: "classifier" | "anomaly_detector";
  loaded: boolean;
  meta: Record<string, unknown>;
}

export interface ModelsListResponse {
  models: ModelInfo[];
  registry: Record<string, unknown>;
}

export interface ActiveModelsResponse {
  classifier: string | null;
  autoencoder: string | null;
  anomaly_threshold: number | null;
}

export interface AnalyticsSummary {
  total_predictions: number;
  avg_processing_time_ms: number;
  avg_anomaly_score: number;
  status_distribution: Record<string, number>;
}

export type FaultDistribution = Record<string, number>;

export interface AnomalyTrendPoint {
  timestamp: string;
  anomaly_score: number;
  status: PredictionStatus;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
  };
}

export interface AnalyzeParams {
  device_id?: string;
  environment?: string;
  operating_condition?: string;
  rpm?: number;
}
