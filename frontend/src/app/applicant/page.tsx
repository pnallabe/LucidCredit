/**
 * app/applicant/page.tsx
 * -----------------------
 * Applicant communication generator.
 *
 * Internal staff enter an application ID to generate a compliance-validated
 * applicant communication (adverse action notice, approval letter, etc.).
 * The page renders:
 *   - Generated email subject + body
 *   - Compliance validation status
 *   - Adverse action notice details (if applicable)
 *   - Citations and confidence badge
 */
"use client";

import * as React from "react";
import { Mail, CheckCircle, XCircle, AlertTriangle, Copy } from "lucide-react";
import { cn } from "@/lib/utils";
import { ConfidenceBadge } from "@/components/ConfidenceBadge";
import { CitationDrawer } from "@/components/CitationDrawer";
import {
  applicantCommunication,
  type ApplicantCommunicationResponse,
} from "@/lib/copilot-client";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ApplicantPage() {
  const [form, setForm] = React.useState({
    applicationId: "",
    source: "thinfile",
    communicationType: "decline",
    channel: "email",
    language: "en",
  });
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState<ApplicantCommunicationResponse | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = React.useState(false);
  const [copied, setCopied] = React.useState<"subject" | "body" | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!form.applicationId.trim()) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await applicantCommunication({
        application_id: form.applicationId.trim(),
        source: form.source,
        communication_type: form.communicationType as "decline" | "approve" | "counteroffer",
        channel: form.channel as "email" | "sms" | "letter",
        language: form.language,
      });
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  }

  async function copyText(field: "subject" | "body") {
    if (!result) return;
    const text = field === "subject" ? result.subject_line : result.body;
    await navigator.clipboard.writeText(text);
    setCopied(field);
    setTimeout(() => setCopied(null), 1500);
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6 px-4 py-8">
      {/* Page title */}
      <div className="flex items-center gap-2">
        <Mail className="h-5 w-5 text-brand-400" />
        <h1 className="text-2xl font-bold text-slate-100">Applicant Communication</h1>
      </div>
      <p className="text-sm text-slate-400">
        Generate a compliance-validated applicant communication. All outputs are checked against
        ECOA/FCRA rules before delivery.
      </p>

      {/* Form */}
      <form
        onSubmit={handleSubmit}
        className="rounded-xl border border-slate-700 bg-slate-800/60 p-5"
      >
        <div className="grid gap-4 sm:grid-cols-2">
          {/* Application ID */}
          <div className="sm:col-span-2">
            <label className="mb-1.5 block text-xs font-medium text-slate-300">
              Application ID <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={form.applicationId}
              onChange={(e) => setForm((f) => ({ ...f, applicationId: e.target.value }))}
              placeholder="e.g. app-3f7a9b2c-0001-4e8d-b123-abcdef012345"
              className="w-full rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30"
              required
            />
          </div>

          {/* Source */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-slate-300">Source System</label>
            <select
              value={form.source}
              onChange={(e) => setForm((f) => ({ ...f, source: e.target.value }))}
              className="w-full rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30"
            >
              <option value="thinfile">ThinFile Engine</option>
              <option value="credit-risk-platform">credit-risk-platform</option>
            </select>
          </div>

          {/* Communication type */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-slate-300">
              Communication Type
            </label>
            <select
              value={form.communicationType}
              onChange={(e) => setForm((f) => ({ ...f, communicationType: e.target.value }))}
              className="w-full rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30"
            >
              <option value="decline">Decline (Adverse Action)</option>
              <option value="approve">Approval</option>
              <option value="counteroffer">Counter-offer</option>
            </select>
          </div>

          {/* Channel */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-slate-300">Channel</label>
            <select
              value={form.channel}
              onChange={(e) => setForm((f) => ({ ...f, channel: e.target.value }))}
              className="w-full rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30"
            >
              <option value="email">Email</option>
              <option value="letter">Letter</option>
              <option value="sms">SMS</option>
            </select>
          </div>

          {/* Language */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-slate-300">Language</label>
            <select
              value={form.language}
              onChange={(e) => setForm((f) => ({ ...f, language: e.target.value }))}
              className="w-full rounded-lg border border-slate-600 bg-slate-900 px-3 py-2 text-sm text-slate-100 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30"
            >
              <option value="en">English</option>
              <option value="es">Spanish</option>
            </select>
          </div>
        </div>

        <div className="mt-4 flex justify-end">
          <button
            type="submit"
            disabled={loading || !form.applicationId.trim()}
            className="flex items-center gap-2 rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-brand-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? (
              <>
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                Generating…
              </>
            ) : (
              <>
                <Mail className="h-4 w-4" />
                Generate Communication
              </>
            )}
          </button>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-800/60 bg-red-950/30 px-4 py-3">
          <XCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-400" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-4 animate-fade-in">
          {/* Compliance status bar */}
          <div
            className={cn(
              "flex items-center gap-3 rounded-lg border px-4 py-3",
              result.compliance_validated
                ? "border-emerald-800/40 bg-emerald-950/20"
                : "border-red-800/40 bg-red-950/20"
            )}
          >
            {result.compliance_validated ? (
              <CheckCircle className="h-4 w-4 text-emerald-400" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-red-400" />
            )}
            <div className="flex-1">
              <p
                className={cn(
                  "text-sm font-medium",
                  result.compliance_validated ? "text-emerald-300" : "text-red-300"
                )}
              >
                {result.compliance_validated
                  ? "ECOA/FCRA compliance validated"
                  : "Compliance validation failed — review required before sending"}
              </p>
            </div>
            <ConfidenceBadge score={result.confidence_score} size="sm" />
          </div>

          {/* Subject line */}
          <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
                Subject
              </span>
              <button
                onClick={() => copyText("subject")}
                className="flex items-center gap-1 text-xs text-slate-500 transition hover:text-slate-300"
              >
                <Copy className="h-3 w-3" />
                {copied === "subject" ? "Copied!" : "Copy"}
              </button>
            </div>
            <p className="text-sm font-medium text-slate-100">{result.subject_line}</p>
          </div>

          {/* Body */}
          <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
                Communication Body
              </span>
              <div className="flex items-center gap-2">
                {result.citations.length > 0 && (
                  <button
                    onClick={() => setDrawerOpen(true)}
                    className="text-xs text-brand-400 transition hover:text-brand-300"
                  >
                    {result.citations.length} source{result.citations.length !== 1 ? "s" : ""}
                  </button>
                )}
                <button
                  onClick={() => copyText("body")}
                  className="flex items-center gap-1 text-xs text-slate-500 transition hover:text-slate-300"
                >
                  <Copy className="h-3 w-3" />
                  {copied === "body" ? "Copied!" : "Copy"}
                </button>
              </div>
            </div>
            <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-slate-200">
              {result.body}
            </pre>
          </div>

          {/* Adverse Action Notice details */}
          {result.adverse_action_notice && (
            <details className="rounded-xl border border-slate-700 bg-slate-800/40">
              <summary className="cursor-pointer px-5 py-3 text-xs font-medium text-slate-400 hover:text-slate-300">
                Adverse Action Notice Details (Reg B / FCRA § 615)
              </summary>
              <div className="space-y-3 px-5 pb-4 pt-1">
                {/* Specific reasons */}
                <div>
                  <p className="mb-1 text-xs font-medium text-slate-400">Specific Reasons</p>
                  {result.adverse_action_notice.specific_reasons.length === 0 ? (
                    <p className="text-xs text-slate-500">No reason codes provided.</p>
                  ) : (
                    <ul className="space-y-1">
                      {result.adverse_action_notice.specific_reasons.map((r, i) => (
                        <li key={i} className="flex items-start gap-2 text-xs text-slate-300">
                          <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-brand-500" />
                          {r}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                {/* FCRA disclosure */}
                {result.adverse_action_notice.fcra_disclosure && (
                  <div className="rounded-lg bg-slate-900/60 p-3">
                    <p className="mb-1 text-xs font-medium text-slate-400">
                      FCRA § 615 Disclosure
                    </p>
                    <p className="text-xs leading-relaxed text-slate-400">
                      {result.adverse_action_notice.fcra_disclosure}
                    </p>
                  </div>
                )}
              </div>
            </details>
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

      <CitationDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        citations={result?.citations ?? []}
      />
    </div>
  );
}
