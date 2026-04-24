/**
 * FeatureBreakdown.tsx
 * --------------------
 * SHAP waterfall chart showing feature importance for a credit decision.
 *
 * Takes a list of { feature, shap_value } records from the CRP API explanation
 * and renders a horizontal bar chart using Recharts.
 *
 * Positive SHAP values → pushing toward approval (green).
 * Negative SHAP values → pushing toward decline (red).
 * Features are sorted by absolute SHAP value (most impactful first).
 */
"use client";

import * as React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { cn } from "@/lib/utils";

export interface ShapFeature {
  feature: string;
  shap_value: number;
  display_name?: string;
  actual_value?: string | number | null;
}

interface FeatureBreakdownProps {
  features: ShapFeature[];
  baseValue?: number;
  title?: string;
  maxFeatures?: number;
  className?: string;
}

interface TooltipPayload {
  payload?: ShapFeature & { fill: string };
}

function ShapTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayload[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  if (!d) return null;

  const positive = (d.shap_value ?? 0) >= 0;
  return (
    <div className="rounded-lg border border-slate-600 bg-slate-800 p-3 shadow-xl">
      <p className="mb-1 text-sm font-medium text-slate-100">
        {d.display_name ?? d.feature}
      </p>
      {d.actual_value != null && (
        <p className="mb-1 text-xs text-slate-400">
          Value: <span className="text-slate-200">{String(d.actual_value)}</span>
        </p>
      )}
      <p className={cn("text-sm font-bold", positive ? "text-emerald-400" : "text-red-400")}>
        SHAP: {positive ? "+" : ""}
        {(d.shap_value ?? 0).toFixed(4)}
      </p>
      <p className="mt-1 text-xs text-slate-500">
        {positive ? "↑ Supports approval" : "↓ Supports decline"}
      </p>
    </div>
  );
}

export function FeatureBreakdown({
  features,
  baseValue,
  title = "Feature Importance (SHAP)",
  maxFeatures = 12,
  className,
}: FeatureBreakdownProps) {
  if (!features || features.length === 0) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded-xl border border-slate-700 bg-slate-800/60 p-8",
          className
        )}
      >
        <p className="text-sm text-slate-500">No SHAP values available for this decision.</p>
      </div>
    );
  }

  // Sort by absolute value, take top N, reverse for waterfall display (lowest at top)
  const sorted = [...features]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, maxFeatures)
    .reverse();

  const data = sorted.map((f) => ({
    ...f,
    abs: Math.abs(f.shap_value),
    label: f.display_name ?? f.feature.replace(/_/g, " "),
  }));

  const maxAbs = Math.max(...data.map((d) => d.abs), 0.01);
  const domainMax = maxAbs * 1.15;

  return (
    <div
      className={cn(
        "rounded-xl border border-slate-700 bg-slate-800/60 backdrop-blur",
        className
      )}
    >
      {/* Header */}
      <div className="border-b border-slate-700 px-5 py-3">
        <h3 className="text-sm font-medium text-slate-200">{title}</h3>
        {baseValue != null && (
          <p className="mt-0.5 text-xs text-slate-400">
            Base score: <span className="text-slate-300 font-mono">{baseValue.toFixed(4)}</span>
          </p>
        )}
      </div>

      {/* Chart */}
      <div className="px-4 pb-4 pt-3">
        <ResponsiveContainer width="100%" height={Math.max(data.length * 36, 180)}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 0, right: 24, left: 8, bottom: 0 }}
            barCategoryGap="20%"
          >
            <CartesianGrid
              strokeDasharray="3 3"
              horizontal={false}
              stroke="rgba(148,163,184,0.1)"
            />
            <XAxis
              type="number"
              domain={[-domainMax, domainMax]}
              tick={{ fill: "#94a3b8", fontSize: 10 }}
              tickFormatter={(v: number) => v.toFixed(3)}
              axisLine={{ stroke: "rgba(148,163,184,0.2)" }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={140}
              tick={{ fill: "#cbd5e1", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<ShapTooltip />} cursor={{ fill: "rgba(148,163,184,0.05)" }} />
            <ReferenceLine x={0} stroke="rgba(148,163,184,0.4)" strokeWidth={1} />
            <Bar dataKey="shap_value" radius={[0, 3, 3, 0]} maxBarSize={20}>
              {data.map((entry, index) => (
                <Cell
                  key={index}
                  fill={entry.shap_value >= 0 ? "#10b981" : "#ef4444"}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Legend */}
        <div className="mt-2 flex items-center justify-center gap-5 text-xs text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" />
            Supports approval
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-red-500" />
            Supports decline
          </span>
        </div>
      </div>

      {/* Truncation notice */}
      {features.length > maxFeatures && (
        <div className="border-t border-slate-700 px-5 py-2 text-xs text-slate-500">
          Showing top {maxFeatures} of {features.length} features by absolute SHAP value.
        </div>
      )}
    </div>
  );
}
