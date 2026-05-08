/**
 * api.ts
 * -------
 * API client for the eco-material analysis pipeline.
 *
 * Keeps all existing chat management functions (getChats, createChat, etc.)
 * and replaces sendMessage / pollMessage with the full analysis pipeline API.
 *
 * Key design notes
 * ----------------
 * - startAnalysis() uses FormData (multipart), NOT JSON.
 *   Do NOT set Content-Type manually — the browser sets it automatically
 *   with the correct boundary string when you pass a FormData body.
 *   All other POST requests use JSON.
 *
 * - ChatService is a singleton on the backend, so session_id is the stable
 *   identifier for a pipeline run. Store it alongside task_id and pass it
 *   to followup / clearSession calls.
 *
 * - startPolling() returns a cancel function. Always call it on component
 *   unmount or when the user navigates away, to avoid orphaned intervals.
 */

import { auth } from "./firebase";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";
const ANALYSIS_BASE = `${API_URL}/api/v1/analysis`;

// =============================================================================
// Auth helpers — unchanged from original
// =============================================================================

const getHeaders = async (): Promise<Record<string, string>> => {
  const user = auth.currentUser;
  const token = user ? await user.getIdToken() : null;
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
};

/**
 * Headers for multipart/form-data requests.
 * Content-Type is intentionally omitted — the browser sets it automatically
 * with the correct boundary when you pass a FormData body to fetch().
 */
const getMultipartHeaders = async (): Promise<Record<string, string>> => {
  const user = auth.currentUser;
  const token = user ? await user.getIdToken() : null;
  return {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
};

export const logout = async (): Promise<void> => {
  await auth.signOut();
};

// =============================================================================
// Existing chat management — unchanged
// =============================================================================

export interface ChatType {
  id: number;
  title: string;
}

export const getChats = async (): Promise<ChatType[]> => {
  const response = await fetch(`${API_URL}/chats`, {
    headers: await getHeaders(),
  });
  if (!response.ok) throw new Error("Failed to fetch chats");
  return response.json();
};

export const createChat = async (): Promise<{ id: number }> => {
  const response = await fetch(`${API_URL}/chats`, {
    method: "POST",
    headers: await getHeaders(),
  });
  if (!response.ok) throw new Error("Failed to create chat");
  return response.json();
};

export const deleteChat = async (id: number): Promise<void> => {
  const response = await fetch(`${API_URL}/chats/${id}`, {
    method: "DELETE",
    headers: await getHeaders(),
  });
  if (!response.ok) throw new Error("Failed to delete chat");
};

export const updateChatTitle = async (id: number): Promise<ChatType> => {
  const response = await fetch(`${API_URL}/chats/${id}/title`, {
    method: "POST",
    headers: await getHeaders(),
  });
  if (!response.ok) throw new Error("Failed to update chat title");
  return response.json();
};

// =============================================================================
// Pipeline types — mirror backend Pydantic schemas exactly
// =============================================================================

export interface ThoughtType {
  node: string;
  type: "info" | "result" | "warning";
  text: string;
}

export interface MaterialPrediction {
  material_name: string;
  eco_score?: number;
  confidence: number;
  predicted_strength_MPa: number | null;
  predicted_stiffness_GPa: number | null;
  description?: string;
}

export interface ExplanationType {
  /** The three-paragraph explanation text. */
  text: string;
}

export interface ReportType {
  top_recommendation: MaterialPrediction | null;
  alternatives: MaterialPrediction[];
  explanation: ExplanationType | null;
  load_estimate: Record<string, unknown> | null;
  mechanical_requirements: Record<string, unknown> | null;
  geometry_summary: Record<string, unknown> | null;
  inference_confidence: number | null;
  warnings: string[];
  /** Full thought trace from all pipeline nodes, included in the report. */
  all_thoughts: ThoughtType[];
}

export interface LoadEstimate {
  load_type: string;
  primary_stress_mode: string;
  magnitude_range_N: [number, number] | number[];
  safety_factor: number;
  operating_temp_C: [number, number] | number[];
  is_fatigue_critical: boolean;
  deflection_category: string;
  confidence: number;
  reasoning: string;
}

export interface LoadPreview {
  load_estimate: LoadEstimate;
  expected_load_n: number;
  safety_factor: number;
  thoughts: ThoughtType[];
}

export interface StructuralPrimitiveSummary {
  archetype: string;
  effective_length_mm: number;
  cross_section_area_mm2: number;
  wall_thickness_mm: number;
  applied_force_N: number;
  applied_pressure_Pa: number;
  support_condition: string;
  stress_MPa: number;
  required_strength_MPa: number;
  required_E_GPa: number;
  buckling_risk: boolean;
  deflection_mm: number;
  deflection_limit_mm: number;
  governing_formula: string;
  derivation: Record<string, unknown>;
  warnings: string[];
}

export interface MlInputVector {
  required_tensile_strength_MPa: number;
  required_stiffness_GPa: number;
  stiffness_dominated: boolean;
  buckling_risk: boolean;
  min_operating_temp_C: number;
  max_operating_temp_C: number;
  max_density_kg_m3: number;
  fatigue_critical: boolean;
  recyclability_priority: number;
  governing_primitive?: string;
  primitives?: StructuralPrimitiveSummary[];
  derivation?: Record<string, unknown>;
  warnings?: string[];
}

export interface FeaturePreview {
  geometry: Record<string, unknown>;
  load_estimate: LoadEstimate;
  ml_input_vector: MlInputVector;
  interpreted_features: Record<string, unknown>;
}

export interface PipelineError {
  summary: string;
  root_cause: string;
  recommended_action: string;
}

export interface AnalysisResult {
  /** 'complete' | 'followup_answered' | 'error' */
  status: string;
  session_id: string;
  report: ReportType | null;
  followup_answer: string | null;
  warnings: string[];
  error: PipelineError | null;
  thoughts: ThoughtType[];
}

export interface TaskStatus {
  task_id: string;
  /** 'pending' | 'running' | 'complete' | 'failed' */
  status: "pending" | "running" | "complete" | "failed";
  /** Latest status message from the running pipeline node. */
  progress: string | null;
  /** Accumulated reasoning steps — append new items to the frontend display. */
  thoughts: ThoughtType[];
  result: AnalysisResult | null;
  /** Populated when status='failed'. */
  error: string | null;
}

export interface AcceptedResponse {
  status: "accepted";
  task_id: string;
  /** Store this — required for followup and clearSession calls. */
  session_id: string;
  message: string;
}

export interface ClearSessionResponse {
  status: "success" | "already_cleared";
  message: string;
}

export interface CadParserPreview {
  upload_id: string;
  filename: string;
  geometry: Record<string, unknown>;
}

// =============================================================================
// Analysis pipeline API
// =============================================================================

/**
 * Start a new eco-material analysis.
 *
 * Uses multipart/form-data to upload the CAD file alongside the text fields.
 * Returns task_id (for polling) and session_id (for follow-up calls).
 * Store session_id in component state alongside task_id.
 *
 * @param file                  The .step, .stp, or .stl file from an <input type="file">.
 * @param objectDescription     Plain-language part name. e.g. "bicycle crank arm".
 * @param recyclabilityPriority Eco weighting 0.0–1.0. Default 0.7.
 */
export const startAnalysis = async (
  file: File,
  objectDescription: string,
  recyclabilityPriority = 0.7
): Promise<AcceptedResponse> => {
  const form = new FormData();
  form.append("file", file);
  form.append("object_description", objectDescription);
  form.append("recyclability_priority", String(recyclabilityPriority));

  const response = await fetch(`${ANALYSIS_BASE}/start`, {
    method: "POST",
    // No Content-Type header — browser sets multipart/form-data + boundary automatically
    headers: await getMultipartHeaders(),
    body: form,
  });

  if (!response.ok) {
    const detail = await _extractErrorDetail(response);
    throw new Error(`Failed to start analysis: ${detail}`);
  }

  return response.json();
};

export const parseCadFile = async (file: File): Promise<CadParserPreview> => {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${ANALYSIS_BASE}/parse`, {
    method: "POST",
    headers: await getMultipartHeaders(),
    body: form,
  });

  if (!response.ok) {
    const detail = await _extractErrorDetail(response);
    throw new Error(`Failed to parse CAD file: ${detail}`);
  }

  return response.json();
};

export const continueAnalysis = async (
  uploadId: string,
  objectDescription: string,
  expectedLoadN?: number,
  safetyFactor?: number
): Promise<AcceptedResponse> => {
  const response = await fetch(`${ANALYSIS_BASE}/continue`, {
    method: "POST",
    headers: await getHeaders(),
    body: JSON.stringify({
      upload_id: uploadId,
      object_description: objectDescription,
      recyclability_priority: 0.7,
      expected_load_n: expectedLoadN,
      safety_factor: safetyFactor,
    }),
  });

  if (!response.ok) {
    const detail = await _extractErrorDetail(response);
    throw new Error(`Failed to continue analysis: ${detail}`);
  }

  return response.json();
};

export const previewFeatureTranslation = async (
  uploadId: string,
  objectDescription: string,
  expectedLoadN: number,
  safetyFactor: number
): Promise<FeaturePreview> => {
  const response = await fetch(`${ANALYSIS_BASE}/feature-preview`, {
    method: "POST",
    headers: await getHeaders(),
    body: JSON.stringify({
      upload_id: uploadId,
      object_description: objectDescription,
      expected_load_n: expectedLoadN,
      safety_factor: safetyFactor,
    }),
  });

  if (!response.ok) {
    const detail = await _extractErrorDetail(response);
    throw new Error(`Failed to translate features: ${detail}`);
  }

  return response.json();
};

export const previewLoadAssumptions = async (
  uploadId: string,
  objectDescription: string
): Promise<LoadPreview> => {
  const response = await fetch(`${ANALYSIS_BASE}/load-preview`, {
    method: "POST",
    headers: await getHeaders(),
    body: JSON.stringify({
      upload_id: uploadId,
      object_description: objectDescription,
    }),
  });

  if (!response.ok) {
    const detail = await _extractErrorDetail(response);
    throw new Error(`Failed to infer load assumptions: ${detail}`);
  }

  return response.json();
};

/**
 * Ask a follow-up question about a completed recommendation.
 *
 * The pipeline does not re-run. ExplanationNode answers directly from the
 * stored pipeline context. Returns a new task_id to poll for the answer.
 *
 * @param sessionId The session_id from the completed analysis.
 * @param question  The engineer's follow-up question.
 */
export const askFollowup = async (
  sessionId: string,
  question: string
): Promise<AcceptedResponse> => {
  const response = await fetch(`${ANALYSIS_BASE}/followup`, {
    method: "POST",
    headers: await getHeaders(),
    body: JSON.stringify({
      session_id: sessionId,
      question,
    }),
  });

  if (!response.ok) {
    const detail = await _extractErrorDetail(response);
    throw new Error(`Failed to submit follow-up: ${detail}`);
  }

  return response.json();
};

/**
 * Poll the current status of a pipeline task.
 *
 * Prefer startPolling() for most UI use cases — it handles the polling loop,
 * terminal state detection, and cleanup automatically.
 * Use this directly if you need manual control over the polling loop.
 */
export const pollTaskStatus = async (taskId: string): Promise<TaskStatus> => {
  const response = await fetch(`${ANALYSIS_BASE}/status/${taskId}`, {
    headers: await getHeaders(),
  });

  if (response.status === 404) {
    throw new Error(`Task '${taskId}' not found.`);
  }
  if (!response.ok) {
    const detail = await _extractErrorDetail(response);
    throw new Error(`Failed to poll task status: ${detail}`);
  }

  return response.json();
};

/**
 * Invalidate a session. Blocks future followup on this session_id.
 * Task history remains readable via pollTaskStatus().
 *
 * @param sessionId The session_id to invalidate.
 */
export const clearSession = async (
  sessionId: string
): Promise<ClearSessionResponse> => {
  const response = await fetch(`${ANALYSIS_BASE}/${sessionId}`, {
    method: "DELETE",
    headers: await getHeaders(),
  });

  if (!response.ok) {
    const detail = await _extractErrorDetail(response);
    throw new Error(`Failed to clear session: ${detail}`);
  }

  return response.json();
};

// =============================================================================
// Polling helper
// =============================================================================

export interface PollingCallbacks {
  /**
   * Fires on every successful poll while the task is pending or running.
   * Use this to update a progress bar, status message, and thoughts list.
   */
  onUpdate: (taskStatus: TaskStatus) => void;

  /**
   * Fires when status='complete' (analysis finished or followup answered).
   * result.status distinguishes 'complete' from 'followup_answered'.
   */
  onComplete: (taskStatus: TaskStatus) => void;

  /**
   * Fires when status='failed'.
   * error contains the AbortReason from the pipeline supervisor.
   */
  onError: (error: string, taskStatus: TaskStatus) => void;
}

/**
 * Start polling a task until it reaches a terminal state.
 *
 * Handles all terminal states (complete, failed)
 * and cleans up the interval automatically.
 *
 * Returns a cancel function — call it on component unmount or user navigation
 * to stop polling without waiting for a terminal state.
 *
 * @param taskId    The task_id returned by startAnalysis or askFollowup.
 * @param callbacks Object with onUpdate, onComplete, onError.
 * @param intervalMs Polling interval in milliseconds. Default 1500.
 *
 * @example
 * const cancel = startPolling(task_id, {
 *   onUpdate: (s) => setProgress(s.progress),
 *   onComplete: (s) => setReport(s.result?.report),
 *   onError: (e, s) => setError(e),
 * });
 * // In cleanup: cancel();
 */
export const startPolling = (
  taskId: string,
  callbacks: PollingCallbacks,
  intervalMs = 3000
): (() => void) => {
  let cancelled = false;
  let consecutiveErrors = 0;
  const MAX_CONSECUTIVE_ERRORS = 3;

  const poll = async () => {
    if (cancelled) return;

    try {
      const taskStatus = await pollTaskStatus(taskId);
      consecutiveErrors = 0; // reset on success

      if (cancelled) return;

      switch (taskStatus.status) {
        case "pending":
        case "running":
          callbacks.onUpdate(taskStatus);
          break;

        case "complete":
          callbacks.onUpdate(taskStatus);
          callbacks.onComplete(taskStatus);
          cancelled = true; // stop polling — terminal state
          break;

        case "failed":
          callbacks.onUpdate(taskStatus);
          callbacks.onError(
            taskStatus.error ?? "An unexpected error occurred.",
            taskStatus
          );
          cancelled = true; // stop polling — terminal state
          break;

        default:
          // Unknown status — treat as still running
          callbacks.onUpdate(taskStatus);
      }
    } catch (err) {
      consecutiveErrors += 1;
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        cancelled = true;
        callbacks.onError(
          `Lost connection to server after ${MAX_CONSECUTIVE_ERRORS} failed attempts. `
          + (err instanceof Error ? err.message : String(err)),
          {
            task_id: taskId,
            status: "failed",
            progress: null,
            thoughts: [],
            result: null,
            error: "Connection lost",
          }
        );
      }
      // Otherwise: swallow the error and retry on the next interval
    }
  };

  // Fire immediately, then on interval
  poll();
  const interval = setInterval(poll, intervalMs);

  // Cancel function
  return () => {
    cancelled = true;
    clearInterval(interval);
  };
};

// =============================================================================
// Internal helpers
// =============================================================================

/**
 * Extract a human-readable error message from a failed fetch response.
 * FastAPI returns { detail: string | ValidationError[] } on errors.
 */
const _extractErrorDetail = async (response: Response): Promise<string> => {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      // Pydantic validation errors: [{loc, msg, type}]
      return body.detail
        .map((e: { loc: string[]; msg: string }) => `${e.loc.join(".")}: ${e.msg}`)
        .join("; ");
    }
    return JSON.stringify(body);
  } catch {
    return `HTTP ${response.status} ${response.statusText}`;
  }
};
