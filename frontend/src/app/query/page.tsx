/**
 * app/query/page.tsx
 * ------------------
 * LucidCredit v2 — Conversational analyst chat.
 *
 * Natural chat interface with real-time streaming, conversation history,
 * and contextual follow-up suggestions. Clarifications handled conversationally.
 */
"use client";

import * as React from "react";
import { Send, Terminal, RefreshCw, Sparkles } from "lucide-react";
import { streamChat } from "@/lib/copilot-client";
import { ChartBlock, type ChartSpec } from "@/components/ChartBlock";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  followUps?: string[];
}

const STARTERS = [
  "What is the current delinquency rate across the portfolio?",
  "Show the charge-off rate trend by quarter.",
  "What is the approval rate by FICO tier for personal loans?",
  "What is the total outstanding balance by product type?",
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function QueryPage() {
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [input, setInput] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [sessionId, setSessionId] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const abortRef = React.useRef<(() => void) | null>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const bottomRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  React.useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [input]);

  function handleNewConversation() {
    abortRef.current?.();
    setMessages([]);
    setSessionId(null);
    setError(null);
    setInput("");
    setLoading(false);
  }

  function handleSend(text?: string) {
    const message = (text ?? input).trim();
    if (!message || loading) return;

    setInput("");
    setError(null);

    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: message };
    const assistantMsgId = crypto.randomUUID();
    const assistantMsg: Message = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      streaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setLoading(true);

    abortRef.current?.();
    abortRef.current = streamChat(message, sessionId, null, {
      onChunk: (chunk) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId ? { ...m, content: m.content + chunk } : m
          )
        );
      },
      onDone: (sid) => {
        setSessionId(sid);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, streaming: false, followUps: getFollowUps(message) }
              : m
          )
        );
        setLoading(false);
      },
      onError: (err) => {
        setError(err);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  streaming: false,
                  content: m.content || "Something went wrong. Please try again.",
                }
              : m
          )
        );
        setLoading(false);
      },
    });
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSend();
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-700/60 px-6 py-3">
        <div className="flex items-center gap-2">
          <Terminal className="h-4 w-4 text-brand-400" />
          <h1 className="text-base font-semibold text-slate-100">Analyst Copilot</h1>
          {sessionId && (
            <span className="rounded-full bg-slate-800 px-2 py-0.5 font-mono text-[10px] text-slate-500">
              {sessionId.slice(0, 8)}&hellip;
            </span>
          )}
        </div>
        {!isEmpty && (
          <button
            onClick={handleNewConversation}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-slate-400 transition hover:border-slate-600 hover:text-slate-200"
          >
            <RefreshCw className="h-3 w-3" />
            New chat
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        {isEmpty ? (
          <div className="mx-auto max-w-2xl">
            <div className="mb-8 text-center">
              <div className="mb-3 inline-flex h-12 w-12 items-center justify-center rounded-full bg-brand-950/60 ring-1 ring-brand-800/40">
                <Sparkles className="h-6 w-6 text-brand-400" />
              </div>
              <h2 className="text-xl font-semibold text-slate-100">LucidCredit</h2>
              <p className="mt-1 text-sm text-slate-400">
                Ask anything about the portfolio — delinquency, balances, charge-offs,
                approvals, trends, or credit risk strategy.
              </p>
            </div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {STARTERS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSend(s)}
                  className="rounded-xl border border-slate-700 bg-slate-800/60 px-4 py-3 text-left text-sm text-slate-300 transition hover:border-brand-700/60 hover:bg-slate-800 hover:text-slate-100"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-2xl space-y-6">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} onFollowUp={handleSend} />
            ))}
            {error && (
              <p className="rounded-lg border border-red-800/50 bg-red-950/30 px-4 py-2.5 text-sm text-red-300">
                {error}
              </p>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-slate-700/60 bg-slate-900/80 px-4 py-4 backdrop-blur">
        <div className="mx-auto max-w-2xl">
          <div className="flex items-end gap-2 rounded-xl border border-slate-700 bg-slate-800/60 px-4 py-3 focus-within:border-brand-600/60">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about the portfolio..."
              disabled={loading}
              rows={1}
              className="flex-1 resize-none bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none"
            />
            <button
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-brand-600 text-white transition hover:bg-brand-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? (
                <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
              ) : (
                <Send className="h-3.5 w-3.5" />
              )}
            </button>
          </div>
          <p className="mt-1.5 text-center text-[11px] text-slate-600">
            Cmd+Enter to send &middot; Conversation remembered across turns
          </p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ChatMessage({
  message,
  onFollowUp,
}: {
  message: Message;
  onFollowUp: (text: string) => void;
}) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-brand-700/30 px-4 py-2.5 text-sm text-slate-100 ring-1 ring-brand-600/20">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <div className="mt-1 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-brand-900/60 ring-1 ring-brand-700/40">
        <Sparkles className="h-3.5 w-3.5 text-brand-400" />
      </div>
      <div className="flex-1 space-y-3">
        <div className="prose prose-sm prose-invert max-w-none text-slate-300">
          <MarkdownText text={message.content} />
          {message.streaming && (
            <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-brand-400 align-middle" />
          )}
        </div>
        {!message.streaming && (message.followUps?.length ?? 0) > 0 && (
          <div className="flex flex-wrap gap-2 pt-1">
            {message.followUps!.map((s) => (
              <button
                key={s}
                onClick={() => onFollowUp(s)}
                className="rounded-full border border-slate-700 bg-slate-800/60 px-3 py-1 text-xs text-slate-400 transition hover:border-brand-600/60 hover:text-slate-200"
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function MarkdownText({ text }: { text: string }) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let listBuffer: string[] = [];
  let codeBuffer: string[] = [];
  let chartBuffer: string[] = [];
  let inCode = false;
  let inChart = false;

  function flushList() {
    if (!listBuffer.length) return;
    elements.push(
      <ul key={elements.length} className="my-2 list-disc space-y-0.5 pl-5">
        {listBuffer.map((l, i) => (
          <li key={i}>
            <InlineMarkdown text={l} />
          </li>
        ))}
      </ul>
    );
    listBuffer = [];
  }

  function flushChart() {
    if (!chartBuffer.length) return;
    try {
      const spec: ChartSpec = JSON.parse(chartBuffer.join("\n"));
      elements.push(<ChartBlock key={elements.length} spec={spec} />);
    } catch {
      // Malformed JSON — skip silently (chart data is still streaming in)
    }
    chartBuffer = [];
  }

  function flushCode() {
    if (!codeBuffer.length) return;
    elements.push(
      <pre
        key={elements.length}
        className="my-2 overflow-x-auto rounded-lg bg-slate-900 p-3 font-mono text-xs text-slate-300"
      >
        {codeBuffer.join("\n")}
      </pre>
    );
    codeBuffer = [];
  }

  for (const line of lines) {
    // ── chart block ──
    if (!inCode && !inChart && line.startsWith("```chart")) {
      flushList();
      inChart = true;
      continue;
    }
    if (inChart) {
      if (line === "```") {
        flushChart();
        inChart = false;
      } else {
        chartBuffer.push(line);
      }
      continue;
    }
    // ── regular code block ──
    if (line.startsWith("```")) {
      if (inCode) {
        flushCode();
        inCode = false;
      } else {
        flushList();
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeBuffer.push(line);
      continue;
    }
    if (line.startsWith("- ") || line.startsWith("* ")) {
      listBuffer.push(line.slice(2));
      continue;
    }
    flushList();
    if (line.startsWith("### ")) {
      elements.push(
        <h4 key={elements.length} className="mt-3 mb-0.5 text-sm font-semibold text-slate-200">
          {line.slice(4)}
        </h4>
      );
    } else if (line.startsWith("## ")) {
      elements.push(
        <h3 key={elements.length} className="mt-4 mb-1 text-base font-semibold text-slate-100">
          {line.slice(3)}
        </h3>
      );
    } else if (line.startsWith("# ")) {
      elements.push(
        <h2 key={elements.length} className="mt-4 mb-1 text-lg font-bold text-slate-100">
          {line.slice(2)}
        </h2>
      );
    } else if (!line.trim()) {
      elements.push(<div key={elements.length} className="h-2" />);
    } else {
      elements.push(
        <p key={elements.length} className="text-sm leading-relaxed">
          <InlineMarkdown text={line} />
        </p>
      );
    }
  }
  flushList();
  flushCode();
  flushChart();
  return <>{elements}</>;
}

function InlineMarkdown({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return (
            <strong key={i} className="font-semibold text-slate-100">
              {part.slice(2, -2)}
            </strong>
          );
        }
        if (part.startsWith("`") && part.endsWith("`")) {
          return (
            <code key={i} className="rounded bg-slate-800 px-1 font-mono text-xs text-brand-300">
              {part.slice(1, -1)}
            </code>
          );
        }
        return <React.Fragment key={i}>{part}</React.Fragment>;
      })}
    </>
  );
}

function getFollowUps(question: string): string[] {
  const q = question.toLowerCase();
  if (/delinquency|dpd|past\.due/.test(q))
    return [
      "How does this compare to last year?",
      "Break down by product type.",
      "What is driving the trend?",
    ];
  if (/charge\.off|write\.off/.test(q))
    return ["Show the quarterly trend.", "Which vintage has the highest charge-off rate?"];
  if (/approval|origination/.test(q))
    return ["What is the average FICO for approved loans?", "Show origination volume by month."];
  if (/balance|outstanding/.test(q))
    return ["Show the balance trend over 12 months.", "Break down by credit grade."];
  return ["Show me the trend over time.", "Break this down by product type."];
}
