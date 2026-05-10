/**
 * ReasoningTraceCard.tsx
 * ----------------------
 * Collapsible panel that exposes how the AI arrived at its conclusion:
 *   - Retrieval method badge (BigQuery SQL / Domain Knowledge / Vector Search)
 *   - Retrieved data sources with content snippets
 *   - Raw LLM analysis (before citation grounding) — toggled separately
 *   - Suppressed claims count (if any)
 */
"use client";

import * as React from "react";
import { Brain, ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ReasoningTrace, ReasoningContextItem } from "@/lib/copilot-client";

// ---------------------------------------------------------------------------
// Retrieval method badge
// ---------------------------------------------------------------------------

const METHOD_STYLES: Record<string, { label: string; cls: string }> = {
  bigquery_query: {
    label: "BigQuery SQL",
    cls: "bg-blue-900/40 text-blue-300 border-blue-700",
  },
  domain_knowledge: {
    label: "Domain Knowledge",
    cls: "bg-amber-900/40 text-amber-300 border-amber-700",
  },
  vector_search: {
    label: "Vector Search",
    cls: "bg-purple-900/40 text-purple-300 border-purple-700",
  },
  portfolio_api: {
    label: "Portfolio API",
    cls: "bg-green-900/40 text-green-300 border-green-700",
  },
};

function MethodBadge({ method }: { method: string }) {
  const entry = METHOD_STYLES[method] ?? {
    label: method,
    cls: "bg-slate-800 text-slate-400 border-slate-700",
  };
  return (
    <span className={cn("rounded-full border px-2.5 py-0.5 text-xs font-medium", entry.cls)}>
      {entry.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Source type chip
// ---------------------------------------------------------------------------

const SOURCE_STYLES: Record<string, { label: string; cls: string }> = {
  db: { label: "BigQuery", cls: "bg-blue-900/50 text-blue-300 border-blue-700" },
  vector_doc: { label: "Policy Doc", cls: "bg-purple-900/50 text-purple-300 border-purple-700" },
  api: { label: "Portfolio API", cls: "bg-green-900/50 text-green-300 border-green-700" },
  domain_knowledge: {
    label: "Domain Knowledge",
    cls: "bg-amber-900/50 text-amber-300 border-amber-700",
  },
};

function SourceChip({ type }: { type: string }) {
  const entry = SOURCE_STYLES[type] ?? {
    label: type,
    cls: "bg-slate-800 text-slate-400 border-slate-700",
  };
  return (
    <span className={cn("rounded-full border px-2 py-0.5 text-[10px] font-medium", entry.cls)}>
      {entry.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Retrieved context item row
// ---------------------------------------------------------------------------

function ContextRow({ item }: { item: ReasoningContextItem }) {
  // Strip internal prefixes from source_ref for readability
  const label = item.source_ref
    .replace(/^sql_query:/, "SQL: ")
    .replace(/^analytics:/, "")
    .slice(0, 100);

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <SourceChip type={item.source_type} />
        <span className="truncate font-mono text-xs text-slate-500">{label}</span>
      </div>
      <p className="line-clamp-3 text-xs leading-relaxed text-slate-400">{item.snippet}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main card
// ---------------------------------------------------------------------------

interface ReasoningTraceCardProps {
  trace: ReasoningTrace;
}

export function ReasoningTraceCard({ trace }: ReasoningTraceCardProps) {
  const [open, setOpen] = React.useState(false);
  const [rawOpen, setRawOpen] = React.useState(false);

  const hasContext = trace.retrieved_context.length > 0;
  const hasRaw = !!trace.raw_analysis?.trim();
  const suppressedCount = trace.suppressed_claims?.length ?? 0;

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800/40">
      {/* Collapsible header */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-5 py-3 text-left transition hover:bg-slate-800/70 rounded-xl"
      >
        <Brain className="h-4 w-4 flex-shrink-0 text-brand-400" />
        <span className="text-sm font-medium text-slate-300">Reasoning Trace</span>
        <span className="ml-1 text-xs text-slate-500">— how I got here</span>

        {trace.retrieval_method && (
          <span className="ml-auto mr-2">
            <MethodBadge method={trace.retrieval_method} />
          </span>
        )}

        {open ? (
          <ChevronDown className="h-4 w-4 flex-shrink-0 text-slate-500" />
        ) : (
          <ChevronRight className="h-4 w-4 flex-shrink-0 text-slate-500" />
        )}
      </button>

      {/* Expanded body */}
      {open && (
        <div className="space-y-4 border-t border-slate-700 px-5 py-4">
          {/* Retrieved context sources */}
          {hasContext && (
            <div>
              <p className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-500">
                Data Consulted ({trace.retrieved_context.length} source
                {trace.retrieved_context.length !== 1 ? "s" : ""})
              </p>
              <div className="space-y-2">
                {trace.retrieved_context.map((item, idx) => (
                  <ContextRow key={idx} item={item} />
                ))}
              </div>
            </div>
          )}

          {/* Raw LLM analysis */}
          {hasRaw && (
            <div>
              <button
                onClick={() => setRawOpen((o) => !o)}
                className="flex items-center gap-1.5 text-xs font-medium text-slate-400 transition hover:text-slate-200"
              >
                {rawOpen ? (
                  <ChevronDown className="h-3.5 w-3.5" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5" />
                )}
                Raw LLM Analysis
                <span className="ml-1 text-slate-600">(before grounding)</span>
              </button>
              {rawOpen && (
                <pre className="mt-2 max-h-96 overflow-y-auto overflow-x-hidden whitespace-pre-wrap rounded-lg bg-slate-900 p-3 font-mono text-xs leading-relaxed text-slate-300">
                  {trace.raw_analysis}
                </pre>
              )}
            </div>
          )}

          {/* Suppressed claims */}
          {suppressedCount > 0 && (
            <div className="rounded-lg border border-amber-900/40 bg-amber-950/20 px-3 py-2">
              <p className="text-xs font-medium text-amber-400">
                {suppressedCount} claim{suppressedCount !== 1 ? "s" : ""} removed during grounding
              </p>
              <ul className="mt-1 space-y-0.5">
                {trace.suppressed_claims.map((c, i) => (
                  <li key={i} className="text-xs text-slate-500">
                    •{" "}
                    {c.claim_text ??
                      c.sentence ??
                      (typeof c === "string" ? c : JSON.stringify(c))}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
