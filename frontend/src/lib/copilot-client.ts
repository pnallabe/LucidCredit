/**
 * frontend/src/lib/copilot-client.ts
 * ====================================
 * Typed API client for the LucidCredit Copilot backend.
 *
 * All functions return typed responses or throw structured errors.
 * The streaming function returns a function that sets up an EventSource
 * and calls callbacks for each event type.
 */

// ---------------------------------------------------------------------------
// Base URL
// ---------------------------------------------------------------------------

const API_BASE =
  typeof window !== "undefined"
    ? "/api" // browser — goes through Next.js rewrite proxy
    : (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8090"); // SSR

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export interface Citation {
  claim: string;
  source_type: "api" | "db" | "vector_doc" | string;
  source_ref: string;
  confidence: number;
}

export interface ApiError {
  error: string;
  message: string;
}

// ---------------------------------------------------------------------------
// Explain Decision
// ---------------------------------------------------------------------------

export interface ExplainDecisionRequest {
  decision_id: string;
  source: "credit-risk-platform" | "thinfile" | string;
  audience: "analyst" | "applicant";
  language?: string;
  include_counterfactual?: boolean;
  include_shap_narrative?: boolean;
}

export interface Counterfactual {
  primary_lever: string;
  estimated_score_improvement: string;
  supporting_levers?: string[];
}

export interface ExplainDecisionResponse {
  session_id: string;
  narrative: string;
  citations: Citation[];
  confidence_score: number;
  counterfactual?: Counterfactual | null;
  adverse_action_codes: string[];
  compliance_flags: string[];
  audience: string;
}

export async function explainDecision(
  req: ExplainDecisionRequest
): Promise<ExplainDecisionResponse> {
  const res = await fetch(`${API_BASE}/v1/explain/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err: ApiError = await res.json().catch(() => ({
      error: "FETCH_ERROR",
      message: res.statusText,
    }));
    throw Object.assign(new Error(err.message), { code: err.error, status: res.status });
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Analyst Query (non-streaming)
// ---------------------------------------------------------------------------

export interface ClarificationItem {
  id: string;
  question: string;
  options: string[];
}

export interface ReasoningContextItem {
  source_type: string;
  source_ref: string;
  relevance: string;
  snippet: string;
}

export interface ReasoningTrace {
  retrieval_method: string;
  retrieved_context: ReasoningContextItem[];
  raw_analysis: string;
  suppressed_claims: Array<{ claim_text?: string; sentence?: string; [key: string]: unknown }>;
}

export interface AnalystQueryRequest {
  query: string;
  data_scope?: string[];
  session_id?: string | null;
  sql_query?: string | null;
  clarifications?: Record<string, string> | null;
}

export interface AnalystQueryResponse {
  session_id: string;
  answer: string;
  citations: Citation[];
  sql_queries_executed: string[];
  confidence_score: number;
  follow_up_suggestions: string[];
  needs_clarification?: boolean;
  clarification_items?: ClarificationItem[];
  reasoning_trace?: ReasoningTrace | null;
}

export async function analystQuery(
  req: AnalystQueryRequest
): Promise<AnalystQueryResponse> {
  const res = await fetch(`${API_BASE}/v1/query/analyst`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err: ApiError = await res.json().catch(() => ({
      error: "FETCH_ERROR",
      message: res.statusText,
    }));
    throw Object.assign(new Error(err.message), { code: err.error, status: res.status });
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Analyst Query — SSE Streaming
// ---------------------------------------------------------------------------

export type StreamStep =
  | "started"
  | "intent_classified"
  | "retrieval_complete"
  | "grading_complete"
  | "reasoning_complete"
  | "grounding_complete"
  | "confidence_scored"
  | "compliance_checked"
  | "output_formatted"
  | "session_persisted";

export interface StreamStatusEvent {
  step: StreamStep;
  session_id?: string;
  chunk_count?: number;
  citation_count?: number;
  confidence_score?: number;
}

export interface StreamResultEvent extends AnalystQueryResponse {}

export interface StreamClarificationEvent {
  session_id: string;
  needs_clarification: true;
  clarification_items: ClarificationItem[];
}

export interface StreamCallbacks {
  onStatus?: (event: StreamStatusEvent) => void;
  onResult?: (result: StreamResultEvent) => void;
  onClarification?: (event: StreamClarificationEvent) => void;
  onError?: (error: ApiError) => void;
  onDone?: () => void;
}

/**
 * Connect to GET /v1/query/stream and process SSE events.
 *
 * Returns an `abort` function — call it to close the EventSource.
 *
 * Usage:
 *   const abort = streamAnalystQuery({ query: "..." }, {
 *     onStatus: (e) => setStep(e.step),
 *     onResult: (r) => setAnswer(r.answer),
 *     onDone: () => setLoading(false),
 *   });
 */
export function streamAnalystQuery(
  req: AnalystQueryRequest,
  callbacks: StreamCallbacks
): () => void {
  const params = new URLSearchParams({ query: req.query });
  if (req.data_scope?.length) params.set("data_scope", req.data_scope.join(","));
  if (req.sql_query) params.set("sql_query", req.sql_query);
  if (req.session_id) params.set("session_id", req.session_id);
  if (req.clarifications && Object.keys(req.clarifications).length > 0) {
    params.set("clarifications", JSON.stringify(req.clarifications));
  }

  const url = `${API_BASE}/v1/query/stream?${params.toString()}`;
  const es = new EventSource(url);

  es.addEventListener("status", (e: MessageEvent) => {
    try {
      const data: StreamStatusEvent = JSON.parse(e.data);
      callbacks.onStatus?.(data);
    } catch {}
  });

  es.addEventListener("result", (e: MessageEvent) => {
    try {
      const data: StreamResultEvent = JSON.parse(e.data);
      callbacks.onResult?.(data);
    } catch {}
  });

  es.addEventListener("clarification", (e: MessageEvent) => {
    try {
      const data: StreamClarificationEvent = JSON.parse(e.data);
      callbacks.onClarification?.(data);
    } catch {}
  });

  es.addEventListener("error", (e: MessageEvent) => {
    try {
      const data: ApiError = JSON.parse(e.data);
      callbacks.onError?.(data);
    } catch {
      callbacks.onError?.({ error: "CONNECTION_ERROR", message: "Stream disconnected." });
    }
    es.close();
  });

  es.addEventListener("done", () => {
    es.close();
    callbacks.onDone?.();
  });

  // Native EventSource error (network failure)
  es.onerror = () => {
    callbacks.onError?.({ error: "CONNECTION_ERROR", message: "Stream disconnected." });
    callbacks.onDone?.();
    es.close();
  };

  return () => es.close();
}

// ---------------------------------------------------------------------------
// Chat v2 — Conversational endpoint (replaces streamAnalystQuery for the UI)
// ---------------------------------------------------------------------------

export interface ChatCallbacks {
  /** Called with each streamed text chunk as it arrives */
  onChunk: (text: string) => void;
  /** Called when the stream is complete; receives the final session_id */
  onDone: (sessionId: string) => void;
  /** Called on a network or server error */
  onError: (error: string) => void;
}

/**
 * Stream a conversational message to POST /v1/chat/stream.
 *
 * Returns an abort function — call it to cancel the in-flight request.
 *
 * Usage:
 *   const abort = streamChat("What is the delinquency rate?", sessionId, null, {
 *     onChunk: (t) => setAnswer(a => a + t),
 *     onDone:  (sid) => { setSessionId(sid); setLoading(false); },
 *     onError: (e)   => { setError(e); setLoading(false); },
 *   });
 */
export function streamChat(
  message: string,
  sessionId: string | null,
  clarifications: Record<string, string> | null,
  callbacks: ChatCallbacks
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId, clarifications }),
        signal: controller.signal,
      });

      const newSessionId = res.headers.get("X-Session-Id") || sessionId || crypto.randomUUID();

      if (!res.ok) {
        const err = await res.json().catch(() => ({ message: res.statusText }));
        callbacks.onError((err as { message?: string }).message || res.statusText);
        return;
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") {
            callbacks.onDone(newSessionId);
            return;
          }
          if (data.startsWith("[ERROR]")) {
            callbacks.onError(data.slice(7).trim());
            return;
          }
          // The server escapes newlines as \\n so the SSE frame stays single-line
          callbacks.onChunk(data.replace(/\\n/g, "\n"));
        }
      }

      callbacks.onDone(newSessionId);
    } catch (err: unknown) {
      if ((err as Error).name !== "AbortError") {
        callbacks.onError(String(err));
      }
    }
  })();

  return () => controller.abort();
}

// ---------------------------------------------------------------------------
// Applicant Communication
// ---------------------------------------------------------------------------

export interface ApplicantCommunicationRequest {
  application_id: string;
  source: "thinfile" | "credit-risk-platform" | string;
  communication_type: "decline" | "approve" | "counteroffer";
  channel: "email" | "sms" | "letter";
  language?: string;
}

export interface AdverseActionNotice {
  application_id: string;
  notice_date: string;
  action_taken: string;
  creditor_name: string;
  specific_reasons: string[];
  ecoa_notice: string;
  fcra_disclosure: string | null;
}

export interface ApplicantCommunicationResponse {
  session_id: string;
  subject_line: string;
  body: string;
  adverse_action_notice: AdverseActionNotice | null;
  compliance_validated: boolean;
  citations: Citation[];
  confidence_score: number;
}

export async function applicantCommunication(
  req: ApplicantCommunicationRequest
): Promise<ApplicantCommunicationResponse> {
  const res = await fetch(`${API_BASE}/v1/applicant/communication`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err: ApiError = await res.json().catch(() => ({
      error: "FETCH_ERROR",
      message: res.statusText,
    }));
    throw Object.assign(new Error(err.message), { code: err.error, status: res.status });
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Briefing Generate
// ---------------------------------------------------------------------------

export interface BriefingRequest {
  scope: "portfolio" | "segment" | "decision" | string;
  filters?: Record<string, unknown>;
  sections?: string[];
  audience_role?: string;
}

export interface BriefingResponse {
  session_id: string;
  sections: Record<string, string>;
  citations: Citation[];
  confidence_score: number;
  provider_used: string;
  sql_queries_executed: string[];
}

export async function generateBriefing(req: BriefingRequest): Promise<BriefingResponse> {
  const res = await fetch(`${API_BASE}/v1/briefing/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err: ApiError = await res.json().catch(() => ({
      error: "FETCH_ERROR",
      message: res.statusText,
    }));
    throw Object.assign(new Error(err.message), { code: err.error, status: res.status });
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Audit Record
// ---------------------------------------------------------------------------

export interface AuditCitation {
  citation_id: string;
  claim_text: string;
  source_type: string;
  source_ref: string;
  similarity_score: number | null;
  confidence: number | null;
}

export interface AuditRecord {
  session_id: string;
  query_text: string;
  intent: string;
  audience: string;
  source_system: string | null;
  user_id: string | null;
  retrieved_chunks: unknown;
  rendered_prompt: string | null;
  raw_llm_output: string | null;
  grounded_narrative: string | null;
  confidence_score: number | null;
  suppressed_claims: unknown;
  compliance_flags: unknown;
  provider_model: string;
  provider_fallback_used: boolean;
  context_token_count: number | null;
  output_token_count: number | null;
  created_at: string;
  citations: AuditCitation[];
}

export async function getAuditRecord(sessionId: string): Promise<AuditRecord> {
  const res = await fetch(`${API_BASE}/v1/audit/${sessionId}`);
  if (!res.ok) {
    const err: ApiError = await res.json().catch(() => ({
      error: "FETCH_ERROR",
      message: res.statusText,
    }));
    throw Object.assign(new Error(err.message), { code: err.error, status: res.status });
  }
  return res.json();
}
