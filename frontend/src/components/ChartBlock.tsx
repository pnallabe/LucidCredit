/**
 * components/ChartBlock.tsx
 * --------------------------
 * Professional chart component for the LucidCredit analyst chat.
 *
 * Renders line, bar, or area charts from a structured spec emitted by the
 * backend orchestrator. Includes a collapsible "Sources" panel showing the
 * SQL query and any assumed defaults.
 *
 * Uses Recharts (already a project dependency) with a dark/slate theme.
 */
"use client";

import * as React from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Database, Code2, ChevronDown, ChevronUp, TrendingUp } from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChartYKey {
  key: string;
  label: string;
  color: string;
}

export interface ChartSource {
  label: string;
  type?: string;
  sql?: string;
  note?: string;
}

export interface ChartSpec {
  type: "line" | "bar" | "area";
  title: string;
  x_key: string;
  y_keys: ChartYKey[];
  data: Record<string, unknown>[];
  y_format: "percent" | "currency" | "number";
  row_count?: number;
  sources?: ChartSource[];
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function formatValue(value: number, format: ChartSpec["y_format"]): string {
  if (format === "percent") {
    // Values may already be in % (e.g. 3.2) or decimal (0.032)
    const display = value <= 1.0 && value >= -1.0 ? value * 100 : value;
    return `${display.toFixed(2)}%`;
  }
  if (format === "currency") {
    if (Math.abs(value) >= 1_000_000_000) return `$${(value / 1_000_000_000).toFixed(1)}B`;
    if (Math.abs(value) >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M`;
    if (Math.abs(value) >= 1_000) return `$${(value / 1_000).toFixed(0)}K`;
    return `$${value.toFixed(0)}`;
  }
  if (Math.abs(value) >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(2);
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

const CustomTooltip = ({
  active,
  payload,
  label,
  format,
}: {
  active?: boolean;
  payload?: { name: string; value: number; color: string }[];
  label?: string;
  format: ChartSpec["y_format"];
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-slate-600/60 bg-slate-900 px-3 py-2 shadow-xl">
      <p className="mb-1.5 text-[11px] font-medium text-slate-300">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: p.color }}
          />
          <span className="text-[11px] text-slate-400">{p.name}:</span>
          <span className="text-[11px] font-semibold text-slate-200">
            {formatValue(p.value, format)}
          </span>
        </div>
      ))}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Data normalisation
// ---------------------------------------------------------------------------

/**
 * Ensures every x-axis tick is present for every series.
 * Missing values become `null` so Recharts renders a visible gap
 * instead of connecting across non-overlapping product-type ranges.
 */
function normaliseData(
  data: Record<string, unknown>[],
  xKey: string,
  yKeys: ChartYKey[]
): Record<string, unknown>[] {
  if (data.length === 0) return data;

  // Collect all x values in original order (deduplicated)
  const seen = new Set<unknown>();
  const xValues: unknown[] = [];
  for (const row of data) {
    if (!seen.has(row[xKey])) {
      seen.add(row[xKey]);
      xValues.push(row[xKey]);
    }
  }

  // Build a lookup: xValue → row
  const byX = new Map<unknown, Record<string, unknown>>();
  for (const row of data) {
    const existing = byX.get(row[xKey]) ?? {};
    byX.set(row[xKey], { ...existing, ...row });
  }

  // For each x position, ensure every y-key is present (null when absent)
  return xValues.map((xv) => {
    const row = byX.get(xv) ?? { [xKey]: xv };
    const filled: Record<string, unknown> = { [xKey]: xv };
    for (const yk of yKeys) {
      filled[yk.key] = yk.key in row ? row[yk.key] : null;
    }
    return filled;
  });
}

// ---------------------------------------------------------------------------
// Chart sub-components
// ---------------------------------------------------------------------------

const axisStyle = { fontSize: 10, fill: "#64748b" };
const gridStyle = { stroke: "#1e293b", strokeDasharray: "4 4" };
const legendStyle = { fontSize: 11, color: "#94a3b8" };
const commonMargin = { top: 8, right: 16, left: 4, bottom: 4 };

function renderLineChart(spec: ChartSpec) {
  const data = normaliseData(spec.data, spec.x_key, spec.y_keys);
  return (
    <LineChart data={data} margin={commonMargin}>
      <CartesianGrid {...gridStyle} vertical={false} />
      <XAxis
        dataKey={spec.x_key}
        tick={axisStyle}
        axisLine={false}
        tickLine={false}
        interval="preserveStartEnd"
      />
      <YAxis
        tickFormatter={(v) => formatValue(v, spec.y_format)}
        tick={axisStyle}
        axisLine={false}
        tickLine={false}
        width={56}
      />
      <Tooltip content={<CustomTooltip format={spec.y_format} />} />
      {spec.y_keys.length > 1 && <Legend wrapperStyle={legendStyle} />}
      {spec.y_keys.map((yk) => (
        <Line
          key={yk.key}
          type="monotone"
          dataKey={yk.key}
          name={yk.label}
          stroke={yk.color}
          strokeWidth={2}
          dot={spec.data.length <= 24 ? { r: 3, fill: yk.color } : false}
          activeDot={{ r: 5, strokeWidth: 0 }}
          connectNulls={false}
        />
      ))}
    </LineChart>
  );
}

function renderBarChart(spec: ChartSpec) {
  const data = normaliseData(spec.data, spec.x_key, spec.y_keys);
  return (
    <BarChart data={data} margin={commonMargin}>
      <CartesianGrid {...gridStyle} vertical={false} />
      <XAxis
        dataKey={spec.x_key}
        tick={axisStyle}
        axisLine={false}
        tickLine={false}
        interval={spec.data.length > 12 ? "preserveStartEnd" : 0}
      />
      <YAxis
        tickFormatter={(v) => formatValue(v, spec.y_format)}
        tick={axisStyle}
        axisLine={false}
        tickLine={false}
        width={56}
      />
      <Tooltip content={<CustomTooltip format={spec.y_format} />} />
      {spec.y_keys.length > 1 && <Legend wrapperStyle={legendStyle} />}
      {spec.y_keys.map((yk) => (
        <Bar
          key={yk.key}
          dataKey={yk.key}
          name={yk.label}
          fill={yk.color}
          fillOpacity={0.85}
          radius={[3, 3, 0, 0]}
          maxBarSize={48}
        />
      ))}
    </BarChart>
  );
}

function renderAreaChart(spec: ChartSpec) {
  const data = normaliseData(spec.data, spec.x_key, spec.y_keys);
  return (
    <AreaChart data={data} margin={commonMargin}>
      <defs>
        {spec.y_keys.map((yk) => (
          <linearGradient key={yk.key} id={`grad-${yk.key}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={yk.color} stopOpacity={0.25} />
            <stop offset="95%" stopColor={yk.color} stopOpacity={0.02} />
          </linearGradient>
        ))}
      </defs>
      <CartesianGrid {...gridStyle} vertical={false} />
      <XAxis
        dataKey={spec.x_key}
        tick={axisStyle}
        axisLine={false}
        tickLine={false}
        interval="preserveStartEnd"
      />
      <YAxis
        tickFormatter={(v) => formatValue(v, spec.y_format)}
        tick={axisStyle}
        axisLine={false}
        tickLine={false}
        width={56}
      />
      <Tooltip content={<CustomTooltip format={spec.y_format} />} />
      {spec.y_keys.length > 1 && <Legend wrapperStyle={legendStyle} />}
      {spec.y_keys.map((yk) => (
        <Area
          key={yk.key}
          type="monotone"
          dataKey={yk.key}
          name={yk.label}
          stroke={yk.color}
          strokeWidth={2}
          fill={`url(#grad-${yk.key})`}
          dot={false}
          activeDot={{ r: 5, strokeWidth: 0 }}
          connectNulls={false}
        />
      ))}
    </AreaChart>
  );
}

// ---------------------------------------------------------------------------
// SourcesPanel
// ---------------------------------------------------------------------------

function SourcesPanel({ sources }: { sources: ChartSource[] }) {
  const [openSql, setOpenSql] = React.useState<number | null>(null);

  return (
    <div className="border-t border-slate-700/40 px-4 py-2.5 space-y-2">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-600">
        Sources
      </p>
      {sources.map((src, i) => (
        <div key={i} className="space-y-1">
          <div className="flex items-center gap-1.5">
            <Database className="h-3 w-3 flex-shrink-0 text-slate-500" />
            <span className="text-[11px] text-slate-500">{src.label}</span>
            {src.sql && (
              <button
                onClick={() => setOpenSql(openSql === i ? null : i)}
                className="ml-1 flex items-center gap-0.5 rounded px-1 py-0.5 text-[10px] text-slate-600 transition hover:bg-slate-800 hover:text-slate-400"
              >
                <Code2 className="h-2.5 w-2.5" />
                SQL
                {openSql === i ? (
                  <ChevronUp className="h-2.5 w-2.5" />
                ) : (
                  <ChevronDown className="h-2.5 w-2.5" />
                )}
              </button>
            )}
          </div>
          {src.note && (
            <p className="pl-4 text-[11px] italic text-slate-600">{src.note}</p>
          )}
          {src.sql && openSql === i && (
            <pre className="overflow-x-auto rounded-md bg-slate-950 px-3 py-2 font-mono text-[10px] leading-relaxed text-slate-400">
              {src.sql}
            </pre>
          )}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export function ChartBlock({ spec }: { spec: ChartSpec }) {
  const typeLabel =
    spec.type === "line"
      ? "Trend"
      : spec.type === "area"
        ? "Area Trend"
        : "Comparison";

  return (
    <div className="my-3 overflow-hidden rounded-xl border border-slate-700/60 bg-slate-900/70 shadow-lg">
      {/* Header */}
      <div className="flex items-start justify-between border-b border-slate-700/40 px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 mb-0.5">
            <TrendingUp className="h-3.5 w-3.5 flex-shrink-0 text-indigo-400" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-400/80">
              {typeLabel}
            </span>
          </div>
          <p className="text-sm font-medium leading-snug text-slate-200">{spec.title}</p>
        </div>
        {spec.row_count != null && (
          <span className="ml-3 flex-shrink-0 rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-500">
            {spec.row_count.toLocaleString()} pts
          </span>
        )}
      </div>

      {/* Chart */}
      <div className="px-2 pb-2 pt-4" style={{ height: 240 }}>
        <ResponsiveContainer width="100%" height="100%">
          {spec.type === "bar"
            ? renderBarChart(spec)
            : spec.type === "area"
              ? renderAreaChart(spec)
              : renderLineChart(spec)}
        </ResponsiveContainer>
      </div>

      {/* Sources */}
      {(spec.sources?.length ?? 0) > 0 && (
        <SourcesPanel sources={spec.sources!} />
      )}
    </div>
  );
}
