/**
 * app/analyst/page.tsx
 * ---------------------
 * Analyst dashboard — explain a credit decision.
 *
 * The analyst enters a decision ID and source system, then the page
 * calls POST /v1/explain/decision and renders:
 *   - NarrativeCard (grounded narrative + citations)
 *   - FeatureBreakdown (SHAP waterfall, if available)
 *   - CounterfactualPanel (what would change this decision)
 *   - ConfidenceBadge (inline in NarrativeCard)
 */
"use client";

import * as React from "react";
import { Search, AlertCircle, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { NarrativeCard } from "@/components/NarrativeCard";
import { FeatureBreakdown, type ShapFeature } from "@/components/FeatureBreakdown";
import { CounterfactualPanel } from "@/components/CounterfactualPanel";
import {
  explainDecision,
  type ExplainDecisionResponse,
} from "@/lib/copilot-client";

// ---------------------------------------------------------------------------
// Form state types
// ---------------------------------------------------------------------------

interface FormState {
  decisionId: string;
  source: string;
  includeCounterfactual: boolean;
  includeShap: boolean;
}

interface PageState {
  status: "idle" | "loading" | "result" | "error";
  result: ExplainDecisionResponse | null;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Extract SHAP features from the narrative / decision context
// The CRP API may embed SHAP values in a structured field — here we
// parse them from the narrative as a best-effort fallback.
// ---------------------------------------------------------------------------
function parseShapFromNarrative(narrative: string): ShapFeature[] {
  // Pattern: "feature_name: +0.1234" or "feature_name: -0.1234"
  const re = /([a-zA-Z_][a-zA-Z0-9_ ]+):\s*([+-]?\d+\.\d+)\s+SHAP/g;
  const features: ShapFeature[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(narrative)) !== null) {
    features.push({ feature: m[1].trim(), shap_value: parseFloat(m[2]) });
  }
  return features;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AnalystPage() {
  const [form, setForm] = React.useState<FormState>({
    decisionId: "",
    source: "credit-risk-platform",
    includeCounterfactual: true,
    includeShap: true,
  });
  const [page, setPage] = React.useState<PageState>({
    status: "idle",
    result: null,
    error: null,
  });

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!form.decisionId.trim()) return;

    setPage({ status: "loading", result: null, error: null });

    try {
      const res = await explainDecision({
        decision_id: form.decisionId.trim(),
        source: form.source,
        audience: "analyst",
        include_counterfactual: form.includeCounterfactual,
        include_shap_narrative: form.includeShap,
      });
      setPage({ status: "result", result: res, error: null });
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "An unexpected error occurred.";
      setPage({ status: "error", result: null, error: msg });
    }
  }

  const shapFeatures =
    page.result?.narrative ? parseShapFromNarrative(page.result.narrative) : [];

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-8">
      {/* Page title */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Decision Explanation</h1>
        <p className="mt-1 text-sm text-slate-400">
          Enter a decision ID to generate a grounded, citation-backed explanation.
        </p>
      </div>

      {/* Input form */}
      <form
        onSubmit={handleSubmit}
        className="rounded-xl border border-slate-700 bg-slate-800/60 p-5"
      >
        <div className="grid gap-4 sm:grid-cols-2">
          {/* Decision ID */}
          <div className="sm:col-span-2">
            <label
              htmlFor="decisionId"
              className="mb-1.5 block text-xs font-medium text-slate-300"
            >
              Decision ID <span className="text-red-400">*</span>
            </label>
            <input
              id="decisionId"
              type="text"
              value={form.decisionId}
              onChange={(e) => setForm((f) => ({ ...f, decisionId: e.target.value }))}
              placeholder="e.g. 3f7a9b2c-0001-4e8d-b123-abcdef012345"
              className="w-full rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30"
              required
            />
          </div>

          {/* Source system */}
          <div>
            <label
              htmlFor="source"
              className="mb-1.5 block text-xs font-medium text-slate-300"
            >
              Source System
            </label>
            <select
              id="source"
              value={form.source}
              onChange={(e) => setForm((f) => ({ ...f, source: e.target.value }))}
              className="w-full rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30"
            >
              <option value="credit-risk-platform">credit-risk-platform</option>
              <option value="thinfile">ThinFile Engine</option>
            </select>
          </div>

          {/* Options */}
          <div className="flex flex-col gap-2 justify-center">
            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={form.includeCounterfactual}
                onChange={(e) =>
                  setForm((f) => ({ ...f, includeCounterfactual: e.target.checked }))
                }
                className="rounded border-slate-600 bg-slate-900 text-brand-500 focus:ring-brand-500"
              />
              Include counterfactual
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={form.includeShap}
                onChange={(e) =>
                  setForm((f) => ({ ...f, includeShap: e.target.checked }))
                }
                className="rounded border-slate-600 bg-slate-900 text-brand-500 focus:ring-brand-500"
              />
              Include SHAP narrative
            </label>
          </div>
        </div>

        <div className="mt-4 flex justify-end">
          <button
            type="submit"
            disabled={page.status === "loading" || !form.decisionId.trim()}
            className="flex items-center gap-2 rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-brand-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {page.status === "loading" ? (
              <>
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                Generating…
              </>
            ) : (
              <>
                <Search className="h-4 w-4" />
                Explain Decision
              </>
            )}
          </button>
        </div>
      </form>

      {/* Error state */}
      {page.status === "error" && page.error && (
        <div className="flex items-start gap-3 rounded-lg border border-red-800/60 bg-red-950/30 px-4 py-3">
          <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
          <p className="text-sm text-red-300">{page.error}</p>
        </div>
      )}

      {/* Results */}
      {page.status === "result" && page.result && (
        <div className="space-y-4 animate-fade-in">
          {/* Session metadata */}
          <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
            <span>
              Session:{" "}
              <a
                href={`/audit/${page.result.session_id}`}
                className="font-mono text-brand-300 underline underline-offset-2 hover:text-brand-200"
              >
                {page.result.session_id}
              </a>
            </span>
            <ChevronRight className="h-3 w-3" />
            <span>
              Audience: <span className="text-slate-200">{page.result.audience}</span>
            </span>
            {page.result.adverse_action_codes.length > 0 && (
              <>
                <ChevronRight className="h-3 w-3" />
                <span>
                  Adverse action codes:{" "}
                  <span className="font-mono text-slate-200">
                    {page.result.adverse_action_codes.join(", ")}
                  </span>
                </span>
              </>
            )}
          </div>

          {/* Narrative */}
          <NarrativeCard
            narrative={page.result.narrative}
            citations={page.result.citations}
            confidenceScore={page.result.confidence_score}
            audience="analyst"
          />

          {/* SHAP + Counterfactual grid */}
          <div className="grid gap-4 lg:grid-cols-2">
            <FeatureBreakdown features={shapFeatures} title="SHAP Feature Importance" />
            <CounterfactualPanel
              counterfactual={page.result.counterfactual}
              decisionId={form.decisionId}
              source={form.source}
            />
          </div>
        </div>
      )}
    </div>
  );
}
