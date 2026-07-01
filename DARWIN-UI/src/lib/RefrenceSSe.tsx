/**
DarwinLogViewer.tsx — Complete React reference for Darwin Logger SSE API.

Consumes: http://localhost:8765/stream  (SSE)
Queries:   http://localhost:8765/history?session=&levels=&limit=

Renders every event type Darwin exposes:
  • LogEvent (plain, json, list)
  • ProgressStart/Update/End
  • SpinnerStart/Update/End  
  • SessionStart/End

Uses special log-type chips for json/list payloads so you know how to handle
different data types in your own frontend.
**/

import React, { useState, useEffect, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES — mirrors Darwin's event protocol exactly
// ═══════════════════════════════════════════════════════════════════════════════

export type LogType = "info" | "warning" | "error" | "success";

export interface LogEvent {
  type: "LogEvent";
  message: string;
  log_type: LogType;
  session: string;
  timestamp: number;
  datetime: string;
  payload: any | null;
  format: "" | "json" | "list";
}

export interface ProgressStartEvent {
  type: "ProgressStartEvent";
  id: string;
  total: number;
  label: string;
  session: string;
  timestamp: number;
}

export interface ProgressUpdateEvent {
  type: "ProgressUpdateEvent";
  id: string;
  current: number;
  status: string;
  session: string;
  timestamp: number;
}

export interface ProgressEndEvent {
  type: "ProgressEndEvent";
  id: string;
  session: string;
  timestamp: number;
}

export interface SpinnerStartEvent {
  type: "SpinnerStartEvent";
  id: string;
  message: string;
  session: string;
  timestamp: number;
}

export interface SpinnerUpdateEvent {
  type: "SpinnerUpdateEvent";
  id: string;
  message: string;
  session: string;
  timestamp: number;
}

export interface SpinnerEndEvent {
  type: "SpinnerEndEvent";
  id: string;
  success: boolean;
  session: string;
  timestamp: number;
}

export interface SessionStartEvent {
  type: "SessionStartEvent";
  name: string;
  timestamp: number;
}

export interface SessionEndEvent {
  type: "SessionEndEvent";
  name: string;
  timestamp: number;
}

export type DarwinEvent =
  | LogEvent
  | ProgressStartEvent
  | ProgressUpdateEvent
  | ProgressEndEvent
  | SpinnerStartEvent
  | SpinnerUpdateEvent
  | SpinnerEndEvent
  | SessionStartEvent
  | SessionEndEvent;

// ═══════════════════════════════════════════════════════════════════════════════
// SSE HOOK — reusable, handles reconnect, tracks connection state
// ═══════════════════════════════════════════════════════════════════════════════

interface UseDarwinSSEOptions {
  url?: string;
  onEvent?: (event: DarwinEvent) => void;
}

interface UseDarwinSSEReturn {
  events: DarwinEvent[];
  isConnected: boolean;
  error: string | null;
  reconnect: () => void;
  clear: () => void;
}

function useDarwinSSE(options: UseDarwinSSEOptions = {}): UseDarwinSSEReturn {
  const { url = "http://localhost:8765/stream", onEvent } = options;

  const [events, setEvents] = useState<DarwinEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const esRef = useRef<EventSource | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const connect = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
    }

    const es = new EventSource(url);
    esRef.current = es;
    setError(null);

    es.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    es.onmessage = (e) => {
      try {
        const data: DarwinEvent = JSON.parse(e.data);
        setEvents((prev) => [...prev, data]);
        onEventRef.current?.(data);
      } catch (err) {
        console.error("Failed to parse SSE data:", err);
      }
    };

    es.onerror = () => {
      setIsConnected(false);
      if (es.readyState === EventSource.CLOSED) {
        setError("Connection closed. Reconnecting...");
      }
    };
  }, [url]);

  const reconnect = useCallback(() => {
    connect();
  }, [connect]);

  const clear = useCallback(() => {
    setEvents([]);
  }, []);

  useEffect(() => {
    connect();
    return () => {
      esRef.current?.close();
    };
  }, [connect]);

  return { events, isConnected, error, reconnect, clear };
}

// ═══════════════════════════════════════════════════════════════════════════════
// HISTORY QUERY — fetch past logs via HTTP GET
// ═══════════════════════════════════════════════════════════════════════════════

interface HistoryQuery {
  session?: string;
  levels?: LogType[];
  limit?: number;
}

async function fetchHistory(
  baseUrl = "http://localhost:8765",
  query: HistoryQuery = {}
): Promise<LogEvent[]> {
  const params = new URLSearchParams();
  if (query.session) params.set("session", query.session);
  if (query.levels) params.set("levels", query.levels.join(","));
  if (query.limit) params.set("limit", String(query.limit));

  const res = await fetch(`${baseUrl}/history?${params}`);
  if (!res.ok) throw new Error(`History fetch failed: ${res.status}`);
  return res.json();
}

// ═══════════════════════════════════════════════════════════════════════════════
// RENDERERS — one per event type, so you can copy/paste what you need
// ═══════════════════════════════════════════════════════════════════════════════

const LOG_TYPE_COLORS: Record<LogType, string> = {
  info: "#3b82f6",
  warning: "#f59e0b",
  error: "#ef4444",
  success: "#22c55e",
};

const LOG_TYPE_BG: Record<LogType, string> = {
  info: "rgba(59,130,246,0.1)",
  warning: "rgba(245,158,11,0.1)",
  error: "rgba(239,68,68,0.1)",
  success: "rgba(34,197,94,0.1)",
};

/** Renders a plain log line */
function PlainLog({ event }: { event: LogEvent }) {
  return (
    <div
      style={{
        padding: "6px 12px",
        borderLeft: `3px solid ${LOG_TYPE_COLORS[event.log_type]}`,
        background: LOG_TYPE_BG[event.log_type],
        marginBottom: 2,
        fontFamily: "monospace",
        fontSize: 13,
      }}
    >
      <span style={{ color: "#888", fontSize: 11 }}>
        {new Date(event.timestamp * 1000).toLocaleTimeString()}
      </span>{" "}
      <span
        style={{
          textTransform: "uppercase",
          fontSize: 10,
          fontWeight: 700,
          color: LOG_TYPE_COLORS[event.log_type],
          marginRight: 8,
        }}
      >
        {event.log_type}
      </span>
      {event.session && (
        <span
          style={{
            background: "#333",
            color: "#aaa",
            padding: "1px 6px",
            borderRadius: 4,
            fontSize: 10,
            marginRight: 8,
          }}
        >
          {event.session}
        </span>
      )}
      <span style={{ color: "#e2e8f0" }}>{event.message}</span>
    </div>
  );
}

/** Renders structured JSON payload with __darwin_type__ badge */
function JsonLog({ event }: { event: LogEvent }) {
  const dtype = event.payload?.__darwin_type__ || "dict";
  return (
    <div
      style={{
        padding: "8px 12px",
        borderLeft: `3px solid ${LOG_TYPE_COLORS[event.log_type]}`,
        background: LOG_TYPE_BG[event.log_type],
        marginBottom: 4,
        fontFamily: "monospace",
        fontSize: 13,
      }}
    >
      <div style={{ marginBottom: 4 }}>
        <span style={{ color: "#888", fontSize: 11 }}>
          {new Date(event.timestamp * 1000).toLocaleTimeString()}
        </span>{" "}
        <span
          style={{
            textTransform: "uppercase",
            fontSize: 10,
            fontWeight: 700,
            color: LOG_TYPE_COLORS[event.log_type],
            marginRight: 8,
          }}
        >
          {event.log_type}
        </span>
        {/* SPECIAL: data-type chip so you know this is structured */}
        <span
          style={{
            background: "#7c3aed",
            color: "#fff",
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 10,
            fontWeight: 600,
            marginRight: 8,
          }}
        >
          📦 json
        </span>
        {dtype !== "dict" && (
          <span
            style={{
              background: "#0891b2",
              color: "#fff",
              padding: "2px 8px",
              borderRadius: 4,
              fontSize: 10,
              fontWeight: 600,
              marginRight: 8,
            }}
          >
            🏷️ {dtype}
          </span>
        )}
        {event.session && (
          <span
            style={{
              background: "#333",
              color: "#aaa",
              padding: "1px 6px",
              borderRadius: 4,
              fontSize: 10,
            }}
          >
            {event.session}
          </span>
        )}
      </div>
      <pre
        style={{
          margin: 0,
          padding: 8,
          background: "#0f172a",
          borderRadius: 6,
          color: "#94a3b8",
          fontSize: 12,
          overflowX: "auto",
        }}
      >
        {JSON.stringify(event.payload, null, 2)}
      </pre>
    </div>
  );
}

/** Renders list payload with __darwin_type__: "list" badge */
function ListLog({ event }: { event: LogEvent }) {
  const items = event.payload?.items || [];
  return (
    <div
      style={{
        padding: "8px 12px",
        borderLeft: `3px solid ${LOG_TYPE_COLORS[event.log_type]}`,
        background: LOG_TYPE_BG[event.log_type],
        marginBottom: 4,
        fontFamily: "monospace",
        fontSize: 13,
      }}
    >
      <div style={{ marginBottom: 4 }}>
        <span style={{ color: "#888", fontSize: 11 }}>
          {new Date(event.timestamp * 1000).toLocaleTimeString()}
        </span>{" "}
        <span
          style={{
            textTransform: "uppercase",
            fontSize: 10,
            fontWeight: 700,
            color: LOG_TYPE_COLORS[event.log_type],
            marginRight: 8,
          }}
        >
          {event.log_type}
        </span>
        {/* SPECIAL: list-type chip */}
        <span
          style={{
            background: "#db2777",
            color: "#fff",
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 10,
            fontWeight: 600,
            marginRight: 8,
          }}
        >
          📋 list ({items.length} items)
        </span>
        {event.session && (
          <span
            style={{
              background: "#333",
              color: "#aaa",
              padding: "1px 6px",
              borderRadius: 4,
              fontSize: 10,
            }}
          >
            {event.session}
          </span>
        )}
      </div>
      <pre
        style={{
          margin: 0,
          padding: 8,
          background: "#0f172a",
          borderRadius: 6,
          color: "#94a3b8",
          fontSize: 12,
          overflowX: "auto",
        }}
      >
        {JSON.stringify(event.payload, null, 2)}
      </pre>
    </div>
  );
}

/** Progress bar renderer */
function ProgressBar({
  event,
}: {
  event: ProgressStartEvent | ProgressUpdateEvent | ProgressEndEvent;
}) {
  const [progress, setProgress] = useState<{
    total: number;
    current: number;
    label: string;
    status: string;
    done: boolean;
  } | null>(null);

  useEffect(() => {
    if (event.type === "ProgressStartEvent") {
      setProgress({
        total: event.total,
        current: 0,
        label: event.label,
        status: "",
        done: false,
      });
    } else if (event.type === "ProgressUpdateEvent" && progress) {
      setProgress({ ...progress, current: event.current, status: event.status });
    } else if (event.type === "ProgressEndEvent") {
      setProgress((p) => (p ? { ...p, current: p.total, done: true } : null));
    }
  }, [event]);

  if (!progress) return null;

  const pct = progress.total > 0 ? (progress.current / progress.total) * 100 : 0;

  return (
    <div
      style={{
        padding: "8px 12px",
        background: "#1e293b",
        borderRadius: 6,
        marginBottom: 4,
        fontFamily: "monospace",
        fontSize: 12,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: "#94a3b8" }}>
          ⏳ {progress.label} {progress.status && `— ${progress.status}`}
        </span>
        <span style={{ color: "#64748b", fontSize: 11 }}>
          {progress.current}/{progress.total} ({pct.toFixed(0)}%)
        </span>
      </div>
      <div
        style={{
          width: "100%",
          height: 6,
          background: "#334155",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: progress.done ? "#22c55e" : "#3b82f6",
            borderRadius: 3,
            transition: "width 0.2s ease",
          }}
        />
      </div>
    </div>
  );
}

/** Spinner renderer */
function Spinner({
  event,
}: {
  event: SpinnerStartEvent | SpinnerUpdateEvent | SpinnerEndEvent;
}) {
  const [spinner, setSpinner] = useState<{
    message: string;
    done: boolean;
    success: boolean;
  } | null>(null);

  useEffect(() => {
    if (event.type === "SpinnerStartEvent") {
      setSpinner({ message: event.message, done: false, success: true });
    } else if (event.type === "SpinnerUpdateEvent" && spinner) {
      setSpinner({ ...spinner, message: event.message });
    } else if (event.type === "SpinnerEndEvent") {
      setSpinner((s) =>
        s ? { ...s, done: true, success: event.success } : null
      );
    }
  }, [event]);

  if (!spinner) return null;

  return (
    <div
      style={{
        padding: "6px 12px",
        background: spinner.done
          ? spinner.success
            ? "rgba(34,197,94,0.1)"
            : "rgba(239,68,68,0.1)"
          : "#1e293b",
        borderRadius: 6,
        marginBottom: 4,
        fontFamily: "monospace",
        fontSize: 12,
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      {spinner.done ? (
        <span style={{ fontSize: 14 }}>{spinner.success ? "✅" : "❌"}</span>
      ) : (
        <span style={{ fontSize: 14, animation: "spin 1s linear infinite" }}>⏳</span>
      )}
      <span style={{ color: spinner.done ? "#e2e8f0" : "#94a3b8" }}>
        {spinner.message}
      </span>
    </div>
  );
}

/** Session marker */
function SessionMarker({
  event,
}: {
  event: SessionStartEvent | SessionEndEvent;
}) {
  const isStart = event.type === "SessionStartEvent";
  return (
    <div
      style={{
        padding: "4px 12px",
        background: isStart ? "rgba(59,130,246,0.15)" : "rgba(100,116,139,0.15)",
        borderLeft: `3px solid ${isStart ? "#3b82f6" : "#64748b"}`,
        marginBottom: 4,
        fontFamily: "monospace",
        fontSize: 11,
        color: "#94a3b8",
      }}
    >
      {isStart ? "▶️" : "⏹️"} Session {isStart ? "started" : "ended"}:{" "}
      <strong style={{ color: "#e2e8f0" }}>{event.name}</strong>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT — Darwin Log Viewer
// ═══════════════════════════════════════════════════════════════════════════════

export default function DarwinLogViewer() {
  const { events, isConnected, error, reconnect, clear } = useDarwinSSE();
  const [historySession, setHistorySession] = useState("");
  const [historyLevels, setHistoryLevels] = useState<LogType[]>([]);
  const [historyEvents, setHistoryEvents] = useState<LogEvent[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  // Fetch history on demand
  const loadHistory = useCallback(async () => {
    try {
      const logs = await fetchHistory("http://localhost:8765", {
        session: historySession || undefined,
        levels: historyLevels.length > 0 ? historyLevels : undefined,
        limit: 100,
      });
      setHistoryEvents(logs);
    } catch (e: any) {
      alert(e.message);
    }
  }, [historySession, historyLevels]);

  // Render a single event based on its type
  const renderEvent = (event: DarwinEvent, key: number) => {
    switch (event.type) {
      case "LogEvent":
        if (event.format === "json") return <JsonLog key={key} event={event} />;
        if (event.format === "list") return <ListLog key={key} event={event} />;
        return <PlainLog key={key} event={event} />;

      case "ProgressStartEvent":
      case "ProgressUpdateEvent":
      case "ProgressEndEvent":
        return <ProgressBar key={key} event={event} />;

      case "SpinnerStartEvent":
      case "SpinnerUpdateEvent":
      case "SpinnerEndEvent":
        return <Spinner key={key} event={event} />;

      case "SessionStartEvent":
      case "SessionEndEvent":
        return <SessionMarker key={key} event={event} />;

      default:
        return (
          <div key={key} style={{ color: "#888", padding: 4, fontSize: 11 }}>
            Unknown event: {JSON.stringify(event)}
          </div>
        );
    }
  };

  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "#0f172a",
        color: "#e2e8f0",
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {/* HEADER */}
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid #1e293b",
          display: "flex",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <h1 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>🧬 Darwin Logger</h1>

        {/* Connection status */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            padding: "4px 10px",
            borderRadius: 12,
            background: isConnected ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)",
            fontSize: 12,
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: isConnected ? "#22c55e" : "#ef4444",
              display: "inline-block",
            }}
          />
          {isConnected ? "Connected" : "Disconnected"}
        </div>

        {error && (
          <span style={{ color: "#ef4444", fontSize: 12 }}>{error}</span>
        )}

        <div style={{ flex: 1 }} />

        {/* History query controls */}
        <input
          type="text"
          placeholder="Session filter..."
          value={historySession}
          onChange={(e) => setHistorySession(e.target.value)}
          style={{
            padding: "4px 10px",
            borderRadius: 4,
            border: "1px solid #334155",
            background: "#1e293b",
            color: "#e2e8f0",
            fontSize: 12,
            width: 140,
          }}
        />
        <select
          multiple
          size={1}
          onChange={(e) => {
            const opts = Array.from(e.target.selectedOptions).map((o) => o.value as LogType);
            setHistoryLevels(opts);
          }}
          style={{
            padding: "4px 10px",
            borderRadius: 4,
            border: "1px solid #334155",
            background: "#1e293b",
            color: "#e2e8f0",
            fontSize: 12,
          }}
        >
          <option value="">all levels</option>
          <option value="info">info</option>
          <option value="warning">warning</option>
          <option value="error">error</option>
          <option value="success">success</option>
        </select>
        <button
          onClick={loadHistory}
          style={{
            padding: "4px 12px",
            borderRadius: 4,
            border: "none",
            background: "#3b82f6",
            color: "#fff",
            fontSize: 12,
            cursor: "pointer",
          }}
        >
          📜 Load History
        </button>
        <button
          onClick={reconnect}
          style={{
            padding: "4px 12px",
            borderRadius: 4,
            border: "1px solid #334155",
            background: "#1e293b",
            color: "#e2e8f0",
            fontSize: 12,
            cursor: "pointer",
          }}
        >
          🔄 Reconnect
        </button>
        <button
          onClick={clear}
          style={{
            padding: "4px 12px",
            borderRadius: 4,
            border: "1px solid #334155",
            background: "#1e293b",
            color: "#e2e8f0",
            fontSize: 12,
            cursor: "pointer",
          }}
        >
          🗑️ Clear
        </button>
      </div>

      {/* HISTORY SECTION (if loaded) */}
      {historyEvents.length > 0 && (
        <div
          style={{
            padding: "8px 16px",
            borderBottom: "1px solid #1e293b",
            background: "#1e293b",
          }}
        >
          <div
            style={{
              fontSize: 11,
              color: "#64748b",
              marginBottom: 6,
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <span>
              📜 History ({historyEvents.length} logs)
              {historySession && ` | session: ${historySession}`}
              {historyLevels.length > 0 && ` | levels: ${historyLevels.join(",")}`}
            </span>
            <button
              onClick={() => setHistoryEvents([])}
              style={{
                background: "none",
                border: "none",
                color: "#64748b",
                cursor: "pointer",
                fontSize: 11,
              }}
            >
              dismiss
            </button>
          </div>
          <div style={{ maxHeight: 200, overflowY: "auto" }}>
            {historyEvents.map((e, i) => renderEvent(e, i))}
          </div>
        </div>
      )}

      {/* LIVE EVENT STREAM */}
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 0" }}>
        {events.length === 0 && (
          <div
            style={{
              textAlign: "center",
              padding: 60,
              color: "#475569",
              fontSize: 14,
            }}
          >
            Waiting for events from Darwin...
            <br />
            <code style={{ fontSize: 11 }}>curl http://localhost:8765/stream</code>
          </div>
        )}
        {events.map((e, i) => renderEvent(e, i))}
        <div ref={bottomRef} />
      </div>

      {/* FOOTER */}
      <div
        style={{
          padding: "6px 16px",
          borderTop: "1px solid #1e293b",
          fontSize: 11,
          color: "#475569",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>{events.length} events received</span>
        <span>Darwin SSE @ localhost:8765</span>
      </div>

      {/* CSS for spinner animation */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPORTS — everything you might need in your own app
// ═══════════════════════════════════════════════════════════════════════════════

export {
  useDarwinSSE,
  fetchHistory,
  // renderers for copy/paste into your own components
  PlainLog,
  JsonLog,
  ListLog,
  ProgressBar,
  Spinner,
  SessionMarker,
};

export type {
  LogType,
  LogEvent,
  ProgressStartEvent,
  ProgressUpdateEvent,
  ProgressEndEvent,
  SpinnerStartEvent,
  SpinnerUpdateEvent,
  SpinnerEndEvent,
  SessionStartEvent,
  SessionEndEvent,
  DarwinEvent,
  UseDarwinSSEReturn,
  HistoryQuery,
};