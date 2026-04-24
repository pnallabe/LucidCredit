/**
 * CounterfactualPanel.tsx
 * -----------------------
 * Interactive "what would change this decision?" panel.
 *
 * Displays the primary lever from the CRP API counterfactual response,
 * supporting levers, and estimated score improvement.
 *
 * An "Explore Scenario" form allows the analyst to pose a custom
 * "what if" question which is sent to the explain endpoint
 * with `include_counterfactual: true`.
 */
"use client";

import * as React from "react";
import { ArrowRight, TrendingUp, Lightbulb, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Counterfactual, ExplainDecisionRequest } from "@/lib/copilot-client";
import { explainDecision } from "@/lib/copilot-client";

interface CounterfactualPanelProps {
  counterfactual: Counterfactual | null | undefined;
  decisionId: string;
  source: string;
  className?: string;
}

interface ScenarioState {
  status: "idle" | "loading" | "result" | "error";
  result: Counterfactual | null;
  error: string | null;
}

export function CounterfactualPanel({
  counterfactual,
  decisionId,
  source,
  className,
}: CounterfactualPanelProps) {
  const [scenario, setScenario] = React.useState<ScenarioState>({
    status: "idle",
    result: null,
    error: null,
  });

  const active = scenario.status === "result" ? scenario.result : counterfactual;

  async function handleRefresh() {
    setScenario({ status: "loading", result: null, error: null });
    try {
      const req: ExplainDecisionRequest = {
        decision_id: decisionId,
        source,
        audience: "analyst",
        include_counterfactual: true,
        include_shap_narrative: false,
      };
      const res = await explainDecision(req);
      setScenario({
        status: "result",
        result: res.counterfactual ?? null,
        error: null,
      });
    } catch (err: unknown) {
      setScenario({
        status: "error",
        result: null,
        error: err instanceof Error ? err.message : "Failed to fetch counterfactual.",
      });
    }
  }

  if (!active && scenario.status === "idle") {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center gap-3 rounded-xl border border-slate-700 bg-slate-800/60 p-8",
          className
        )}
      >
        <Lightbulb className="h-8 w-8 text-slate-600" />
        <p className="text-sm text-slate-500">No counterfactual data available.</p>
        <button
          onClick={handleRefresh}
          className="rounded-md bg-brand-700 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-brand-600"
        >
          Generate Counterfactual
        </button>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "rounded-xl border border-slate-700 bg-slate-800/60 backdrop-blur",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-700 px-5 py-3">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-brand-400" />
          <h3 className="text-sm font-medium text-slate-200">Counterfactual Analysis</h3>
        </div>
        <button
          onClick={handleRefresh}
          disabled={scenario.status === "loading"}
          className="flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs text-slate-400 transition hover:bg-slate-700 hover:text-slate-200 disabled:opacity-50"
          title="Refresh counterfactual"
        >
          <RefreshCw
            className={cn("h-3 w-3", scenario.status === "loading" && "animate-spin")}
          />
          Refresh
        </button>
      </div>

      {/* Loading */}
      {scenario.status === "loading" && (
        <div className="flex items-center justify-center py-10">
          <RefreshCw className="h-5 w-5 animate-spin text-brand-400" />
          <span className="ml-2 text-sm text-slate-400">Generating counterfactual…</span>
        </div>
      )}

      {/* Error */}
      {scenario.status === "error" && scenario.error && (
        <div className="px-5 py-4">
          <p className="text-sm text-red-400">{scenario.error}</p>
        </div>
      )}

      {/* Primary lever */}
      {active && scenario.status !== "loading" && (
        <div className="space-y-4 px-5 py-4">
          {/* Primary lever */}
          <div className="rounded-lg bg-brand-950/40 p-4 ring-1 ring-brand-800/40">
            <p className="mb-1 text-xs font-medium uppercase tracking-wider text-brand-400">
              Primary Lever
            </p>
            <div className="flex items-center gap-2">
              <ArrowRight className="h-4 w-4 flex-shrink-0 text-brand-400" />
              <p className="text-sm font-medium text-slate-100">{active.primary_lever}</p>
            </div>
          </div>

          {/* Estimated improvement */}
          {active.estimated_score_improvement && (
            <div className="flex items-center gap-3 rounded-lg border border-emerald-800/40 bg-emerald-950/20 px-4 py-3">
              <TrendingUp className="h-4 w-4 flex-shrink-0 text-emerald-400" />
              <div>
                <p className="text-xs text-emerald-400">Estimated Score Impact</p>
                <p className="text-sm font-semibold text-emerald-300">
                  {active.estimated_score_improvement}
                </p>
              </div>
            </div>
          )}

          {/* Supporting levers */}
          {active.supporting_levers && active.supporting_levers.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-400">
                Additional Actions
              </p>
              <ul className="space-y-1.5">
                {active.supporting_levers.map((lever, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                    <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-brand-500" />
                    {lever}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Note */}
          <p className="text-xs text-slate-500">
            Counterfactual actions are derived from the model's decision boundary.
            They are not guarantees of approval — all applications are subject to full
            underwriting review.
          </p>
        </div>
      )}
    </div>
  );
}
