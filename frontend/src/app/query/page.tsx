/**
 * app/query/page.tsx
 * ------------------
 * Analyst query console with SSE streaming.
 *
 * The analyst types a free-text question. The frontend connects to
 * GET /v1/query/stream and renders:
 *   - Live progress indicator (step-by-step node completion)
 *   - NarrativeCard once the result arrives
 *   - SQL queries executed (if any)
 *   - Follow-up suggestions as clickable pills
 */
"use client";

import * as React from "react";
import {
  Terminal,
  Send,
  CheckCircle,
  Circle,
  AlertCircle,
  Clock,
  Database,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { NarrativeCard } from "@/components/NarrativeCard";
import { ReasoningTraceCard } from "@/components/ReasoningTraceCard";
import {
  streamAnalystQuery,
  type AnalystQueryResponse,
  type ClarificationItem,
  type StreamStep,
  type StreamStatusEvent,
} from "@/lib/copilot-client";

// ---------------------------------------------------------------------------
// Step pipeline definition (controls display order)
// ---------------------------------------------------------------------------

const PIPELINE_STEPS: { step: StreamStep; label: string }[] = [
  { step: "started", label: "Connecting to agent" },
  { step: "intent_classified", label: "Intent classified" },
  { step: "retrieval_complete", label: "Context retrieved" },
  { step: "grading_complete", label: "Documents graded" },
  { step: "reasoning_complete", label: "Reasoning complete" },
  { step: "grounding_complete", label: "Citations enforced" },
  { step: "confidence_scored", label: "Confidence scored" },
  { step: "compliance_checked", label: "Compliance checked" },
  { step: "session_persisted", label: "Session saved" },
];

type StepStatus = "pending" | "active" | "done";

interface StepState {
  status: StepStatus;
  metadata?: StreamStatusEvent;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function QueryPage() {
  const [query, setQuery] = React.useState("");
  const [dataScope, setDataScope] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [steps, setSteps] = React.useState<Record<string, StepState>>({});
  const [result, setResult] = React.useState<AnalystQueryResponse | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [history, setHistory] = React.useState<string[]>([]);
  // Persist the session_id across turns so the backend can recall conversation history
  const [sessionId, setSessionId] = React.useState<string | null>(null);

  // Clarification state
  const [clarificationItems, setClarificationItems] = React.useState<ClarificationItem[]>([]);
  const [clarificationAnswers, setClarificationAnswers] = React.useState<Record<string, string>>({});
  const pendingQueryRef = React.useRef<string>("");

  const abortRef = React.useRef<(() => void) | null>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  function handleNewConversation() {
    setSessionId(null);
    setResult(null);
    setError(null);
    setSteps({});
    setClarificationItems([]);
    setClarificationAnswers({});
    setQuery("");
  }

  function handleSubmit(q?: string, clarifications?: Record<string, string>) {
    const activeQuery = q ?? query;
    if (!activeQuery.trim() || loading) return;

    // Cancel any in-flight stream
    abortRef.current?.();

    setLoading(true);
    setResult(null);
    setError(null);
    setSteps({});
    setClarificationItems([]);
    setClarificationAnswers({});
    if (!clarifications) {
      setHistory((h) => [activeQuery, ...h.slice(0, 9)]);
    }

    abortRef.current = streamAnalystQuery(
      {
        query: activeQuery,
        data_scope: dataScope ? dataScope.split(",").map((s) => s.trim()) : [],
        // Reuse the session_id from the previous turn so the backend can load
        // conversation history via the LangGraph checkpointer.
        session_id: sessionId ?? undefined,
        clarifications: clarifications ?? undefined,
      },
      {
        onStatus: (event) => {
          setSteps((prev) => {
            const updated = { ...prev };
            let found = false;
            for (const { step } of PIPELINE_STEPS) {
              if (step === event.step) {
                updated[step] = { status: "active", metadata: event };
                found = true;
              } else if (!found && !updated[step]) {
                updated[step] = { status: "done" };
              } else if (!found && updated[step]?.status === "active") {
                updated[step] = { ...updated[step], status: "done" };
              }
            }
            return updated;
          });
        },
        onResult: (res) => {
          setResult(res);
          // Capture session_id so subsequent questions in this conversation
          // are linked to the same graph checkpoint (enabling history recall).
          if (res.session_id) setSessionId(res.session_id);
          setSteps((prev) => {
            const updated = { ...prev };
            for (const { step } of PIPELINE_STEPS) {
              updated[step] = { status: "done", metadata: prev[step]?.metadata };
            }
            return updated;
          });
        },
        onClarification: (event) => {
          pendingQueryRef.current = activeQuery;
          setClarificationItems(event.clarification_items);
          setClarificationAnswers(
            Object.fromEntries(event.clarification_items.map((it) => [it.id, ""]))
          );
          setLoading(false);
        },
        onError: (err) => {
          setError(`${err.error}: ${err.message}`);
          setLoading(false);
        },
        onDone: () => {
          setLoading(false);
        },
      }
    );
  }

  function handleClarificationSubmit() {
    // Validate all items have an answer
    const allAnswered = clarificationItems.every((it) => clarificationAnswers[it.id]?.trim());
    if (!allAnswered) return;
    handleSubmit(pendingQueryRef.current, clarificationAnswers);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  }

  // Auto-resize textarea
  React.useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [query]);

  const hasSteps = Object.keys(steps).length > 0;

  return (
    <div className="mx-auto max-w-4xl space-y-6 px-4 py-8">
      {/* Page title */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Terminal className="h-5 w-5 text-brand-400" />
          <h1 className="text-2xl font-bold text-slate-100">Analyst Query Console</h1>
        </div>
        {sessionId && (
          <button
            onClick={handleNewConversation}
            className="rounded-lg border border-slate-600 bg-slate-800 px-3 py-1.5 text-xs text-slate-300 transition hover:border-brand-500 hover:text-brand-300"
          >
            New Conversation
          </button>
        )}
      </div>
      <p className="text-sm text-slate-400">
        Ask anything about credit decisions, portfolio metrics, or regulatory guidance.
        The agent retrieves grounded context and streams its reasoning.
      </p>

      {/* Query input */}
      <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-4">
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. Why was loan application #4321 declined? What is the portfolio DTI distribution for Q1 2026?"
          disabled={loading}
          rows={3}
          className="w-full resize-none bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none"
        />

        <div className="mt-3 flex flex-wrap items-end justify-between gap-3">
          {/* Data scope */}
          <div className="flex items-center gap-2">
            <Database className="h-3.5 w-3.5 text-slate-500" />
            <input
              type="text"
              value={dataScope}
              onChange={(e) => setDataScope(e.target.value)}
              placeholder="Data scope (optional, comma-separated)"
              className="w-52 rounded bg-slate-900 px-2 py-1 text-xs text-slate-300 placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-brand-600"
            />
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">⌘+Enter to send</span>
            <button
              onClick={() => handleSubmit()}
              disabled={loading || !query.trim()}              className="flex items-center gap-1.5 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-brand-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              ) : (
                <Send className="h-4 w-4" />
              )}
              {loading ? "Running…" : "Ask"}
            </button>
          </div>
        </div>
      </div>

      {/* Live pipeline progress */}
      {hasSteps && (
        <div className="rounded-xl border border-slate-700 bg-slate-800/40 px-5 py-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">
            Agent Pipeline
          </p>
          <div className="space-y-2">
            {PIPELINE_STEPS.map(({ step, label }) => {
              const s = steps[step];
              const status: StepStatus = s?.status ?? "pending";
              const meta = s?.metadata;

              return (
                <div key={step} className="flex items-center gap-2.5">
                  {status === "done" && (
                    <CheckCircle className="h-4 w-4 flex-shrink-0 text-emerald-400" />
                  )}
                  {status === "active" && (
                    <Clock className="h-4 w-4 flex-shrink-0 animate-pulse text-brand-400" />
                  )}
                  {status === "pending" && (
                    <Circle className="h-4 w-4 flex-shrink-0 text-slate-600" />
                  )}
                  <span
                    className={cn(
                      "text-sm",
                      status === "done" && "text-slate-300",
                      status === "active" && "font-medium text-brand-300",
                      status === "pending" && "text-slate-600"
                    )}
                  >
                    {label}
                  </span>
                  {/* Step metadata */}
                  {meta && (
                    <span className="ml-auto text-xs text-slate-500">
                      {meta.chunk_count != null && `${meta.chunk_count} chunks`}
                      {meta.citation_count != null && `${meta.citation_count} citations`}
                      {meta.confidence_score != null &&
                        `${Math.round(meta.confidence_score * 100)}% confidence`}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Clarification panel */}
      {clarificationItems.length > 0 && (
        <div className="rounded-xl border border-amber-700/60 bg-amber-950/30 px-5 py-4 space-y-4">
          <p className="text-sm font-medium text-amber-300">
            A bit more context is needed to answer accurately:
          </p>
          {clarificationItems.map((item) => (
            <div key={item.id} className="space-y-2">
              <label className="block text-sm text-slate-200">{item.question}</label>
              {item.options.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {item.options.map((opt) => (
                    <button
                      key={opt}
                      onClick={() =>
                        setClarificationAnswers((prev) => ({ ...prev, [item.id]: opt }))
                      }
                      className={cn(
                        "rounded-full border px-3 py-1.5 text-xs transition",
                        clarificationAnswers[item.id] === opt
                          ? "border-brand-500 bg-brand-950/60 text-brand-200"
                          : "border-slate-600 bg-slate-800 text-slate-300 hover:border-brand-500"
                      )}
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              ) : (
                <input
                  type="text"
                  value={clarificationAnswers[item.id] ?? ""}
                  onChange={(e) =>
                    setClarificationAnswers((prev) => ({ ...prev, [item.id]: e.target.value }))
                  }
                  className="w-full rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30"
                />
              )}
            </div>
          ))}
          <button
            onClick={handleClarificationSubmit}
            disabled={!clarificationItems.every((it) => clarificationAnswers[it.id]?.trim())}
            className="flex items-center gap-1.5 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-brand-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Send className="h-4 w-4" />
            Submit Answer
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-800/60 bg-red-950/30 px-4 py-3">
          <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-4 animate-fade-in">
          <NarrativeCard
            narrative={result.answer}
            citations={result.citations}
            confidenceScore={result.confidence_score}
            audience="analyst"
          />

          {/* Reasoning trace — how the AI got here */}
          {result.reasoning_trace && (
            <ReasoningTraceCard trace={result.reasoning_trace} />
          )}

          {/* SQL queries */}
          {result.sql_queries_executed.length > 0 && (
            <details className="rounded-lg border border-slate-700 bg-slate-800/40">
              <summary className="cursor-pointer px-4 py-2.5 text-xs font-medium text-slate-400 hover:text-slate-300">
                SQL Queries Executed ({result.sql_queries_executed.length})
              </summary>
              <div className="space-y-2 px-4 pb-4">
                {result.sql_queries_executed.map((sql, idx) => (
                  <pre
                    key={idx}
                    className="overflow-x-auto rounded bg-slate-900 p-3 font-mono text-xs text-slate-300"
                  >
                    {sql}
                  </pre>
                ))}
              </div>
            </details>
          )}

          {/* Follow-up suggestions */}
          {result.follow_up_suggestions.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-medium text-slate-400">Follow-up Questions</p>
              <div className="flex flex-wrap gap-2">
                {result.follow_up_suggestions.map((suggestion, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setQuery(suggestion);
                      handleSubmit(suggestion);
                    }}
                    className="rounded-full border border-slate-600 bg-slate-800 px-3 py-1.5 text-xs text-slate-300 transition hover:border-brand-500 hover:bg-brand-950/40 hover:text-brand-300"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Audit link */}
          <p className="text-xs text-slate-500">
            Session:{" "}
            <a
              href={`/audit/${result.session_id}`}
              className="font-mono text-brand-400 underline underline-offset-2 hover:text-brand-300"
            >
              {result.session_id}
            </a>
          </p>
        </div>
      )}

      {/* Query history */}
      {history.length > 0 && !loading && !result && (
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-500">
            Recent Queries
          </p>
          <div className="space-y-1">
            {history.map((q, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setQuery(q);
                  handleSubmit(q);
                }}
                className="w-full truncate rounded-lg border border-slate-700 bg-slate-800/40 px-3 py-2 text-left text-sm text-slate-400 transition hover:border-slate-600 hover:text-slate-300"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
