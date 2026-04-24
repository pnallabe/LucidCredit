/**
 * Root layout — shared chrome for all LucidCredit pages.
 *
 * Structure:
 *   html > body > div.flex
 *     aside (sidebar nav)
 *     main (page content)
 */
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "LucidCredit — AI Credit Copilot",
  description:
    "Grounded, compliance-validated AI explanations for credit decisions.",
};

const NAV_ITEMS = [
  {
    href: "/analyst",
    label: "Explain Decision",
    icon: "✦",
    description: "SHAP + grounded narrative",
  },
  {
    href: "/query",
    label: "Query Console",
    icon: "⌥",
    description: "Streaming analyst Q&A",
  },
  {
    href: "/applicant",
    label: "Applicant Comms",
    icon: "✉",
    description: "ECOA-validated notices",
  },
  {
    href: "/audit",
    label: "Audit Log",
    icon: "◈",
    description: "Session-level audit trail",
  },
] as const;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`dark ${inter.variable}`}>
      <body className="min-h-screen bg-slate-900 font-sans antialiased">
        <div className="flex min-h-screen">
          {/* ── Sidebar ──────────────────────────────────────────────── */}
          <aside className="flex w-60 flex-shrink-0 flex-col border-r border-slate-700/60 bg-slate-900">
            {/* Brand */}
            <div className="border-b border-slate-700/60 px-5 py-5">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600">
                  <span className="text-sm font-bold text-white">LC</span>
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-100">LucidCredit</p>
                  <p className="text-xs text-slate-500">AI Credit Copilot</p>
                </div>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 space-y-0.5 p-3">
              {NAV_ITEMS.map(({ href, label, icon, description }) => (
                <Link
                  key={href}
                  href={href}
                  className="group flex flex-col rounded-lg px-3 py-2.5 text-slate-400 transition hover:bg-slate-800 hover:text-slate-100"
                >
                  <div className="flex items-center gap-2.5">
                    <span className="text-brand-400 text-base leading-none">{icon}</span>
                    <span className="text-sm font-medium">{label}</span>
                  </div>
                  <p className="mt-0.5 pl-7 text-xs text-slate-600 group-hover:text-slate-500">
                    {description}
                  </p>
                </Link>
              ))}
            </nav>

            {/* Footer */}
            <div className="border-t border-slate-700/60 px-5 py-3">
              <p className="text-xs text-slate-600">
                SR 11-7 · ECOA · FCRA § 615
              </p>
              <p className="text-xs text-slate-700">Zero-hallucination RAG</p>
            </div>
          </aside>

          {/* ── Page content ─────────────────────────────────────────── */}
          <main className="flex-1 min-w-0 overflow-y-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
