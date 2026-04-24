/**
 * app/audit/[sessionId]/page.tsx
 * --------------------------------
 * Full audit record detail view.
 *
 * Fetches GET /v1/audit/{sessionId} and renders:
 *   - Session metadata table
 *   - Grounded narrative (NarrativeCard)
 *   - Compliance flags
 *   - Retrieved chunks (collapsible JSON)
 *   - Raw rendered_prompt (collapsible code)
 *   - Raw LLM output (collapsible code)
 *   - Suppressed claims list
 */
"use client";

import * as React from "react";
import { useParams } from "next/navigation";
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  Server,
  FileText,
  Eye,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { NarrativeCard } from "@/components/NarrativeCard";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { getAuditRecord, type AuditRecord, type Citation } from "@/lib/copilot-client";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface ComplianceFlag {
  rule_id: string;
  severity: "error" | "warning" | "info";
  message: string;
}

function asFlags(v: unknown): ComplianceFlag[] {
  return Array.isArray(v) ? (v as ComplianceFlag[]) : [];
}
function asStrings(v: unknown): string[] {
  return Array.isArray(v) ? (v as string[]) : [];
}

function MetaRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <tr className="border-b border-slate-700">
      <td className="py-2.5 pr-4 text-xs font-medium text-slate-400 whitespace-nowrap">{label}</td>
      <td className="py-2.5 text-sm text-slate-200">{value}</td>
    </tr>
  );
}

function CollapsibleCode({
  title,
  content,
  icon: Icon,
}: {
  title: string;
  content: string;
  icon?: React.ComponentType<{ className?: string }>;
}) {
  return (
    <details className="rounded-xl border border-slate-700 bg-slate-800/40">
      <summary className="flex cursor-pointer items-center gap-2 px-5 py-3 text-xs font-medium text-slate-400 hover:text-slate-300">
        {Icon && <Icon className="h-3.5 w-3.5" />}
        {title}
      </summary>
      <pre className="max-h-80 overflow-auto px-5 pb-4 pt-1 font-mono text-xs leading-relaxed text-slate-300">
        {content}
      </pre>
    </details>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AuditPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [record, setRecord] = React.useState<AuditRecord | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!sessionId) return;
    setLoading(true);

    getAuditRecord(sessionId)
      .then(setRecord)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load audit record.");
      })
      .finally(() => setLoading(false));
  }, [sessionId]);

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <span className="h-6 w-6 animate-spin rounded-full border-2 border-brand-500 border-t-transparent" />
        <span className="ml-3 text-sm text-slate-400">Loading audit record…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8">
        <div className="flex items-start gap-3 rounded-lg border border-red-800/60 bg-red-950/30 px-4 py-3">
          <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      </div>
    );
  }

  if (!record) return null;

  const complianceFlags = asFlags(record.compliance_flags);
  const hasViolation = complianceFlags.some((f) => f.severity === "error");

  return (
    <div className="mx-auto max-w-4xl space-y-6 px-4 py-8">
      {/* Page header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-brand-400" />
            <h1 className="text-2xl font-bold text-slate-100">Audit Record</h1>
          </div>
          <p className="mt-1 font-mono text-xs text-slate-500">{sessionId}</p>
        </div>
        <ConfidenceBadge score={record.confidence_score ?? 0} size="lg" showLabel />
      </div>

      {/* Compliance banner */}
      <div
        className={cn(
          "flex items-center gap-3 rounded-lg border px-4 py-3",
          hasViolation
            ? "border-red-800/40 bg-red-950/20"
            : "border-emerald-800/40 bg-emerald-950/20"
        )}
      >
        {hasViolation ? (
          <AlertTriangle className="h-4 w-4 text-red-400" />
        ) : (
          <CheckCircle className="h-4 w-4 text-emerald-400" />
        )}
        <p
          className={cn(
            "text-sm font-medium",
            hasViolation ? "text-red-300" : "text-emerald-300"
          )}
        >
          {hasViolation
            ? `${complianceFlags.filter((f) => f.severity === "error").length} compliance violation(s) detected`
            : "No compliance violations"}
        </p>
      </div>

      {/* Session metadata */}
      <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-5">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-300">
          <Server className="h-4 w-4 text-brand-400" />
          Session Metadata
        </h2>
        <table className="w-full">
          <tbody>
            <MetaRow label="Intent" value={record.intent ?? "—"} />
            <MetaRow label="Audience" value={record.audience} />
            <MetaRow label="Provider / Model" value={record.provider_model} />
            <MetaRow
              label="Fallback Used"
              value={
                record.provider_fallback_used ? (
                  <span className="rounded bg-amber-950/40 px-2 py-0.5 text-xs text-amber-300 ring-1 ring-amber-700/40">
                    Yes
                  </span>
                ) : (
                  "No"
                )
              }
            />
            <MetaRow
              label="Created At"
              value={
                <span className="flex items-center gap-1.5">
                  <Clock className="h-3.5 w-3.5 text-slate-500" />
                  {new Date(record.created_at).toLocaleString()}
                </span>
              }
            />
            <MetaRow
              label="Token Counts"
              value={
                <span className="font-mono text-xs">
                  {record.context_token_count != null || record.output_token_count != null
                    ? `ctx: ${record.context_token_count ?? 0} / out: ${record.output_token_count ?? 0}`
                    : "—"}
                </span>
              }
            />
          </tbody>
        </table>
      </div>

      {/* Grounded narrative */}
      {record.grounded_narrative && (
        <NarrativeCard
          narrative={record.grounded_narrative}
          citations={record.citations.map((c) => ({ claim: c.claim_text, source_type: c.source_type, source_ref: c.source_ref, confidence: c.confidence ?? 0 })) as Citation[]}
          confidenceScore={record.confidence_score ?? 0}
          audience={record.audience as "analyst" | "applicant" | "briefing"}
        />
      )}

      {/* Compliance flags detail */}
      {complianceFlags.length > 0 && (
        <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-5">
          <h2 className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-300">
            <Shield className="h-4 w-4 text-brand-400" />
            Compliance Flags
          </h2>
          <ul className="space-y-2">
            {complianceFlags.map((flag, idx) => (
                <li
                  key={idx}
                  className={cn(
                    "flex items-start gap-3 rounded-lg px-3 py-2.5",
                    flag.severity === "error"
                      ? "bg-red-950/30 ring-1 ring-red-800/40"
                      : flag.severity === "warning"
                      ? "bg-amber-950/30 ring-1 ring-amber-800/40"
                      : "bg-slate-900/40 ring-1 ring-slate-700"
                  )}
                >
                  {flag.severity === "error" ? (
                    <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
                  ) : (
                    <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-400" />
                  )}
                  <div>
                    <span
                      className={cn(
                        "font-mono text-xs font-medium",
                        flag.severity === "error" ? "text-red-400" : "text-amber-400"
                      )}
                    >
                      {flag.rule_id}
                    </span>
                    <p className="text-sm text-slate-300">{flag.message}</p>
                  </div>
                </li>
              )
            )}
          </ul>
        </div>
      )}

      {/* Suppressed claims */}
      {asStrings(record.suppressed_claims).length > 0 && (
        <div className="rounded-xl border border-amber-800/40 bg-amber-950/20 p-5">
          <h2 className="mb-3 flex items-center gap-2 text-sm font-medium text-amber-300">
            <AlertTriangle className="h-4 w-4" />
            Suppressed Claims ({asStrings(record.suppressed_claims).length})
          </h2>
          <ul className="space-y-1.5">
            {asStrings(record.suppressed_claims).map((claim, idx) => (
              <li key={idx} className="flex items-start gap-2 text-sm text-amber-200/80">
                <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-amber-400" />
                {claim}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Retrieved chunks */}
      {record.retrieved_chunks != null && (
        <CollapsibleCode
          title={`Retrieved Chunks (${Array.isArray(record.retrieved_chunks) ? record.retrieved_chunks.length : "?"})`}
          content={JSON.stringify(record.retrieved_chunks, null, 2)}
          icon={Eye}
        />
      )}

      {/* Rendered prompt */}
      {record.rendered_prompt && (
        <CollapsibleCode
          title="Rendered Prompt"
          content={record.rendered_prompt}
          icon={FileText}
        />
      )}

      {/* Raw LLM output */}
      {record.raw_llm_output && (
        <CollapsibleCode
          title="Raw LLM Output"
          content={record.raw_llm_output}
          icon={Server}
        />
      )}
    </div>
  );
}
