/**
 * ConfidenceBadge.tsx
 * -------------------
 * Color-coded badge displaying the session confidence score.
 *
 * Score bands:
 *   >= 0.90  → green   "High Confidence"
 *   0.75–0.89 → amber  "Medium Confidence"
 *   < 0.75   → red    "Low Confidence"  (below threshold — typically blocked)
 */
"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface ConfidenceBadgeProps {
  score: number;
  showLabel?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

function getBand(score: number): {
  label: string;
  color: string;
  bg: string;
  ring: string;
} {
  if (score >= 0.9) {
    return {
      label: "High Confidence",
      color: "text-emerald-400",
      bg: "bg-emerald-950/60",
      ring: "ring-emerald-700",
    };
  }
  if (score >= 0.75) {
    return {
      label: "Medium Confidence",
      color: "text-amber-400",
      bg: "bg-amber-950/60",
      ring: "ring-amber-700",
    };
  }
  return {
    label: "Low Confidence",
    color: "text-red-400",
    bg: "bg-red-950/60",
    ring: "ring-red-700",
  };
}

export function ConfidenceBadge({
  score,
  showLabel = false,
  size = "md",
  className,
}: ConfidenceBadgeProps) {
  const band = getBand(score);
  const pct = Math.round(score * 100);

  const sizeClasses = {
    sm: "text-xs px-2 py-0.5 gap-1.5",
    md: "text-sm px-2.5 py-1 gap-2",
    lg: "text-base px-3 py-1.5 gap-2.5",
  };

  const dotSize = {
    sm: "w-1.5 h-1.5",
    md: "w-2 h-2",
    lg: "w-2.5 h-2.5",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full font-medium ring-1 ring-inset",
        band.bg,
        band.color,
        band.ring,
        sizeClasses[size],
        className
      )}
      title={`Confidence score: ${pct}% — ${band.label}`}
    >
      <span
        className={cn("rounded-full", band.color.replace("text-", "bg-"), dotSize[size])}
        aria-hidden
      />
      <span>{pct}%</span>
      {showLabel && <span className="opacity-70">· {band.label}</span>}
    </span>
  );
}
