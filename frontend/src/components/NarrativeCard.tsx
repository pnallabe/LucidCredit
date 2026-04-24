/**
 * NarrativeCard.tsx
 * -----------------
 * Renders a grounded AI narrative with inline citation superscripts.
 *
 * Each sentence that maps to a citation gets a numbered [n] superscript.
 * Clicking any superscript opens the CitationDrawer scrolled to that citation.
 *
 * The card also shows:
 *   - Confidence badge (top-right)
 *   - Suppressed claims warning (if any)
 *   - SR 11-7 disclosure footer (auto-detected from narrative text)
 */
"use client";

import * as React from "react";
import { AlertTriangle, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { CitationDrawer } from "./CitationDrawer";
import type { Citation } from "@/lib/copilot-client";

// The SR 11-7 footer delimiter inserted by the backend
const SR117_DELIMITER = "─";

interface NarrativeCardProps {
  narrative: string;
  citations: Citation[];
  confidenceScore: number;
  suppressedClaims?: unknown[] | null;
  audience?: "analyst" | "applicant" | "briefing";
  className?: string;
}

/**
 * Split the narrative into the main body and the SR 11-7 disclosure footer.
 * The footer starts at the first line of ─── separators.
 */
function splitNarrative(text: string): { body: string; disclosure: string | null } {
  const idx = text.indexOf(SR117_DELIMITER.repeat(10));
  if (idx === -1) return { body: text, disclosure: null };
  return {
    body: text.slice(0, idx).trim(),
    disclosure: text.slice(idx).trim(),
  };
}

/**
 * Annotate narrative sentences with citation superscripts.
 *
 * Strategy: for each citation whose `claim` text appears in the narrative,
 * append a [n] superscript immediately after the matching phrase's sentence end.
 * This is intentionally simple — a more precise approach would require
 * span-level matching from the backend.
 */
function AnnotatedNarrative({
  body,
  citations,
  onCitationClick,
}: {
  body: string;
  citations: Citation[];
  onCitationClick: (index: number) => void;
}) {
  // Split into paragraphs, then annotate
  const paragraphs = body.split(/\n\n+/);

  return (
    <div className="space-y-4">
      {paragraphs.map((para, pIdx) => {
        // For each paragraph, find citations that match it and inject superscripts
        let annotated: React.ReactNode = para;
        const matchedIndices: number[] = [];

        citations.forEach((cit, cIdx) => {
          if (cit.claim && para.includes(cit.claim.slice(0, 30))) {
            matchedIndices.push(cIdx);
          }
        });

        if (matchedIndices.length > 0) {
          return (
            <p key={pIdx} className="leading-relaxed text-slate-200">
              {para}
              {matchedIndices.map((cIdx) => (
                <button
                  key={cIdx}
                  onClick={() => onCitationClick(cIdx)}
                  className="ml-0.5 inline-flex h-4 w-4 cursor-pointer items-center justify-center rounded-full bg-brand-600 align-super text-[9px] font-bold text-white transition hover:bg-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-400"
                  title={`Citation ${cIdx + 1}: ${citations[cIdx].source_ref}`}
                  aria-label={`View citation ${cIdx + 1}`}
                >
                  {cIdx + 1}
                </button>
              ))}
            </p>
          );
        }

        return (
          <p key={pIdx} className="leading-relaxed text-slate-200">
            {annotated}
          </p>
        );
      })}
    </div>
  );
}

export function NarrativeCard({
  narrative,
  citations,
  confidenceScore,
  suppressedClaims,
  audience = "analyst",
  className,
}: NarrativeCardProps) {
  const [drawerOpen, setDrawerOpen] = React.useState(false);
  const [activeCitation, setActiveCitation] = React.useState<number | null>(null);

  const { body, disclosure } = splitNarrative(narrative);
  const suppressedCount = Array.isArray(suppressedClaims) ? suppressedClaims.length : 0;

  function handleCitationClick(index: number) {
    setActiveCitation(index);
    setDrawerOpen(true);
  }

  return (
    <>
      <div
        className={cn(
          "rounded-xl border border-slate-700 bg-slate-800/60 backdrop-blur",
          className
        )}
      >
        {/* Card header */}
        <div className="flex items-center justify-between border-b border-slate-700 px-5 py-3">
          <div className="flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-brand-400" />
            <span className="text-sm font-medium text-slate-200">
              {audience === "applicant" ? "Your Decision Summary" : "Analysis Narrative"}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <ConfidenceBadge score={confidenceScore} showLabel />
            {citations.length > 0 && (
              <button
                onClick={() => {
                  setActiveCitation(null);
                  setDrawerOpen(true);
                }}
                className="rounded-md px-2.5 py-1 text-xs text-brand-300 transition hover:bg-brand-950/60 hover:text-brand-200"
              >
                {citations.length} source{citations.length !== 1 ? "s" : ""}
              </button>
            )}
          </div>
        </div>

        {/* Suppressed claims warning */}
        {suppressedCount > 0 && (
          <div className="flex items-start gap-2 border-b border-amber-900/40 bg-amber-950/20 px-5 py-2.5">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-amber-400" />
            <p className="text-xs text-amber-300">
              {suppressedCount} claim{suppressedCount !== 1 ? "s were" : " was"} suppressed due to
              insufficient grounding evidence and removed from this narrative.
            </p>
          </div>
        )}

        {/* Narrative body */}
        <div className="px-5 py-4">
          <AnnotatedNarrative
            body={body}
            citations={citations}
            onCitationClick={handleCitationClick}
          />
        </div>

        {/* SR 11-7 disclosure footer (analyst only) */}
        {disclosure && (
          <details className="border-t border-slate-700">
            <summary className="cursor-pointer select-none px-5 py-2.5 text-xs text-slate-500 transition hover:text-slate-400">
              ▸ SR 11-7 Model Risk Disclosure
            </summary>
            <pre className="overflow-x-auto whitespace-pre-wrap px-5 pb-4 font-mono text-xs leading-relaxed text-slate-500">
              {disclosure}
            </pre>
          </details>
        )}
      </div>

      <CitationDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        citations={citations}
        activeCitationIndex={activeCitation}
      />
    </>
  );
}
