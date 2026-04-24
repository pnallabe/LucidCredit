/**
 * CitationDrawer.tsx
 * ------------------
 * Slide-over panel showing citation details for a grounded claim.
 *
 * Opens when the user clicks a [n] superscript inside NarrativeCard.
 * Displays source type, source ref, similarity score, and confidence.
 */
"use client";

import * as React from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { X, Database, FileText, Globe } from "lucide-react";
import { cn } from "@/lib/utils";
import { ConfidenceBadge } from "./ConfidenceBadge";
import type { Citation } from "@/lib/copilot-client";

interface CitationDrawerProps {
  open: boolean;
  onClose: () => void;
  citations: Citation[];
  activeCitationIndex?: number | null;
}

function SourceIcon({ sourceType }: { sourceType: string }) {
  if (sourceType === "db") return <Database className="w-4 h-4 text-brand-400" />;
  if (sourceType === "api") return <Globe className="w-4 h-4 text-brand-400" />;
  return <FileText className="w-4 h-4 text-brand-400" />;
}

function SourceTypePill({ type }: { type: string }) {
  const labels: Record<string, string> = {
    api: "API",
    db: "Database",
    vector_doc: "Policy Doc",
  };
  return (
    <span className="inline-flex items-center gap-1 rounded bg-slate-700/60 px-2 py-0.5 text-xs text-slate-300 ring-1 ring-slate-600">
      <SourceIcon sourceType={type} />
      {labels[type] ?? type}
    </span>
  );
}

export function CitationDrawer({
  open,
  onClose,
  citations,
  activeCitationIndex,
}: CitationDrawerProps) {
  // Scroll to active citation when drawer opens
  const activeRef = React.useRef<HTMLDivElement | null>(null);
  React.useEffect(() => {
    if (open && activeRef.current) {
      activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [open, activeCitationIndex]);

  return (
    <Dialog.Root open={open} onOpenChange={(v) => !v && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm data-[state=open]:animate-fade-in" />
        <Dialog.Content
          className={cn(
            "fixed right-0 top-0 z-50 flex h-full w-full max-w-md flex-col",
            "bg-slate-900 shadow-2xl ring-1 ring-slate-700",
            "data-[state=open]:animate-fade-in"
          )}
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-slate-700 px-5 py-4">
            <div>
              <Dialog.Title className="text-base font-semibold text-slate-100">
                Citations
              </Dialog.Title>
              <p className="mt-0.5 text-xs text-slate-400">
                {citations.length} source{citations.length !== 1 ? "s" : ""} grounding this narrative
              </p>
            </div>
            <Dialog.Close asChild>
              <button
                className="rounded p-1.5 text-slate-400 transition hover:bg-slate-700 hover:text-slate-100"
                aria-label="Close citations"
              >
                <X className="h-4 w-4" />
              </button>
            </Dialog.Close>
          </div>

          {/* Citation list */}
          <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
            {citations.length === 0 ? (
              <p className="py-8 text-center text-sm text-slate-500">No citations available.</p>
            ) : (
              citations.map((citation, idx) => {
                const isActive = activeCitationIndex === idx;
                return (
                  <div
                    key={idx}
                    ref={isActive ? activeRef : undefined}
                    className={cn(
                      "rounded-lg border p-4 transition",
                      isActive
                        ? "border-brand-500 bg-brand-950/40 ring-1 ring-brand-500/40"
                        : "border-slate-700 bg-slate-800/50"
                    )}
                  >
                    {/* Citation number + source type */}
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-brand-600 text-xs font-bold text-white">
                        {idx + 1}
                      </span>
                      <SourceTypePill type={citation.source_type} />
                    </div>

                    {/* Claim text */}
                    <p className="mb-2 text-sm leading-relaxed text-slate-200">
                      "{citation.claim}"
                    </p>

                    {/* Source reference */}
                    <div className="mb-3 rounded bg-slate-700/40 px-3 py-2">
                      <p className="text-xs font-medium text-slate-400">Source</p>
                      <p className="mt-0.5 break-all font-mono text-xs text-brand-300">
                        {citation.source_ref}
                      </p>
                    </div>

                    {/* Scores */}
                    <div className="flex items-center gap-3">
                      <ConfidenceBadge score={citation.confidence} size="sm" showLabel={false} />
                      <span className="text-xs text-slate-400">
                        confidence
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Footer note */}
          <div className="border-t border-slate-700 px-5 py-3">
            <p className="text-xs text-slate-500">
              Every claim in this narrative is traceable to a retrieved source.
              Claims without sufficient grounding are suppressed before delivery.
            </p>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
