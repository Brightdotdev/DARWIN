// DarwinStream.client.tsx
// Usage: point <DarwinStream url="http://localhost:8765/stream" /> at the UnifiedSSEServer

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";

// ─── Schemas (mirror the Python event protocol) ─────────────────────

type LogLevel = "debug" | "info" | "success" | "warning" | "error";

interface LogEvent {
  type: "LogEvent";
  message: string;
  log_type: LogLevel;
  session: string;
  timestamp: number;
  datetime?: string;
  payload?: unknown;
  format?: "json" | "list" | "";
}

interface ProgressStartEvent {
  type: "ProgressStartEvent";
  id: string;
  total: number;
  label: string;
  session: string;
  timestamp: number;
}

interface ProgressUpdateEvent {
  type: "ProgressUpdateEvent";
  id: string;
  current: number;
  status: string;
  session: string;
  timestamp: number;
}

interface ProgressEndEvent {
  type: "ProgressEndEvent";
  id: string;
  session: string;
  timestamp: number;
}

interface SpinnerStartEvent {
  type: "SpinnerStartEvent";
  id: string;
  message: string;
  session: string;
  timestamp: number;
}

interface SpinnerUpdateEvent {
  type: "SpinnerUpdateEvent";
  id: string;
  message: string;
  session: string;
  timestamp: number;
}

interface SpinnerEndEvent {
  type: "SpinnerEndEvent";
  id: string;
  success: boolean;
  session: string;
  timestamp: number;
}

interface SessionStartEvent {
  type: "SessionStartEvent";
  name: string;
  timestamp: number;
}

interface SessionEndEvent {
  type: "SessionEndEvent";
  name: string;
  timestamp: number;
}

type DarwinEvent =
  | LogEvent
  | ProgressStartEvent
  | ProgressUpdateEvent
  | ProgressEndEvent
  | SpinnerStartEvent
  | SpinnerUpdateEvent
  | SpinnerEndEvent
  | SessionStartEvent
  | SessionEndEvent;

// ─── Helpers ────────────────────────────────────────────────────────

const LOG_ICON: Record<LogLevel, string> = {
  debug: "🐛",
  info: "ℹ️",
  success: "✅",
  warning: "⚠️",
  error: "❌",
};

const LOG_COLOR: Record<LogLevel, string> = {
  debug: "#888",
  info: "#58a6ff",
  success: "#3fb950",
  warning: "#d29922",
  error: "#f85149",
};

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

// ─── useDarwinStream hook ───────────────────────────────────────────

interface StreamState {
  connected: boolean;
  events: DarwinEvent[];
  sessions: Map<string, { start: number; logCount: number }>;
  progress: Map<string, { current: number; total: number; label: string }>;
  spinners: Set<string>;
}

export function useDarwinStream(url: string) {
  const [state, setState] = useState<StreamState>({
    connected: false,
    events: [],
    sessions: new Map(),
    progress: new Map(),
    spinners: new Set(),
  });
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const es = new EventSource(url);
    esRef.current = es;

    es.onopen = () => setState((s) => ({ ...s, connected: true }));

    es.onmessage = (msg) => {
      // SSE keepalives are empty lines
      if (!msg.data || msg.data.trim() === "") return;

      let raw: Record<string, unknown>;
      try {
        raw = JSON.parse(msg.data);
      } catch {
        return;
      }

      const event = raw as DarwinEvent;
      setState((prev) => {
        const next = { ...prev, events: [...prev.events, event] };

        // ── Track sessions ──
        if (event.type === "SessionStartEvent") {
          const sessions = new Map(next.sessions);
          sessions.set(event.name, { start: event.timestamp, logCount: 0 });
          return { ...next, sessions };
        }
        if (event.type === "SessionEndEvent") {
          // session remains in map with final state
          return next;
        }

        // ── Track progress bars ──
        if (event.type === "ProgressStartEvent") {
          const progress = new Map(next.progress);
          progress.set(event.id, {
            current: 0,
            total: event.total,
            label: event.label,
          });
          return { ...next, progress };
        }
        if (event.type === "ProgressUpdateEvent") {
          const progress = new Map(next.progress);
          const existing = progress.get(event.id);
          if (existing) {
            progress.set(event.id, { ...existing, current: event.current });
          }
          return { ...next, progress };
        }
        if (event.type === "ProgressEndEvent") {
          const progress = new Map(next.progress);
          progress.delete(event.id);
          return { ...next, progress };
        }

        // ── Track spinners ──
        if (event.type === "SpinnerStartEvent") {
          const spinners = new Set(next.spinners);
          spinners.add(event.id);
          return { ...next, spinners };
        }
        if (event.type === "SpinnerEndEvent") {
          const spinners = new Set(next.spinners);
          spinners.delete(event.id);
          return { ...next, spinners };
        }

        // ── Track log count per session ──
        if (
          event.type === "LogEvent" &&
          event.session &&
          next.sessions.has(event.session)
        ) {
          const sessions = new Map(next.sessions);
          const s = sessions.get(event.session)!;
          sessions.set(event.session, { ...s, logCount: s.logCount + 1 });
          return { ...next, sessions };
        }

        return next;
      });
    };

    es.onerror = () => {
      setState((s) => ({ ...s, connected: false }));
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [url]);

  const clear = useCallback(() => {
    setState({
      connected: false,
      events: [],
      sessions: new Map(),
      progress: new Map(),
      spinners: new Set(),
    });
  }, []);

  return { ...state, clear };
}

// ─── Renderers ───────────────────────────────────────────────────────

function LogLine({ event }: { event: LogEvent }) {
  const icon = LOG_ICON[event.log_type] ?? " ";
  const color = LOG_COLOR[event.log_type] ?? "#fff";
  const tag = event.session ? `[${event.session}]` : "";

  if (event.payload && event.format === "json") {
    return (
      <div style={{ color, fontFamily: "monospace", margin: "2px 0" }}>
        <span>{icon}</span>{" "}
        <span style={{ opacity: 0.6, fontSize: "0.8em" }}>{tag}</span>{" "}
        <StructuredView data={event.payload} />
      </div>
    );
  }

  if (event.payload && event.format === "list") {
    return (
      <div style={{ color, fontFamily: "monospace", margin: "2px 0" }}>
        <span>{icon}</span>{" "}
        <span style={{ opacity: 0.6, fontSize: "0.8em" }}>{tag}</span>{" "}
        <span>{event.message}</span>
        <ListView data={event.payload} />
      </div>
    );
  }

  return (
    <div style={{ color, fontFamily: "monospace", margin: "2px 0" }}>
      <span>{icon}</span>{" "}
      <span style={{ opacity: 0.6, fontSize: "0.8em" }}>
        {formatTime(event.timestamp)} {tag}
      </span>{" "}
      <span>{event.message}</span>
    </div>
  );
}

// ─── Type renderer registry ─────────────────────────────────────────
// Register a React component for any __darwin_type__ marker.
// Unknown types fall back to the generic JSON tree view.

type DarwinTypeRenderer = React.ComponentType<{ data: Record<string, unknown> }>;

class DarwinTypeRegistry {
  private _map = new Map<string, DarwinTypeRenderer>();

  register(typeName: string, component: DarwinTypeRenderer): void {
    this._map.set(typeName, component);
  }

  get(typeName: string): DarwinTypeRenderer | undefined {
    return this._map.get(typeName);
  }

  has(typeName: string): boolean {
    return this._map.has(typeName);
  }
}

// Global default registry — extend it at app init:
//   darwinTypes.register("User", UserCard);
//   darwinTypes.register("Address", AddressBlock);
export const darwinTypes = new DarwinTypeRegistry();

// ─── StructuredView with registry-aware rendering ───────────────────

function StructuredView({
  data,
  registry,
}: {
  data: unknown;
  registry?: DarwinTypeRegistry;
}) {
  const ctxRegistry = useDarwinTypeRegistry();
  registry = registry ?? ctxRegistry;
  if (typeof data !== "object" || data === null) {
    return <span>{String(data)}</span>;
  }

  const dict = data as Record<string, unknown>;
  const darwinType = dict["__darwin_type__"] as string | undefined;

  // ── Built-in types ──
  if (darwinType === "list") {
    const items = (dict["items"] ?? []) as unknown[];
    return (
      <details>
        <summary>list ({items.length} items)</summary>
        {items.map((item, i) => (
          <div key={i} style={{ paddingLeft: 16 }}>
            <StructuredView data={item} registry={registry} />
          </div>
        ))}
      </details>
    );
  }

  if (darwinType === "max_depth" || darwinType === "cycle") {
    return (
      <span style={{ fontStyle: "italic", opacity: 0.5 }}>
        {darwinType}
      </span>
    );
  }

  // ── Registered custom renderer wins over generic JSON ──
  if (darwinType && registry.has(darwinType)) {
    const Renderer = registry.get(darwinType)!;
    return <Renderer data={dict} />;
  }

  // ── Dataclass with no registered renderer — generic tree fallback ──
  if (darwinType) {
    return (
      <details>
        <summary style={{ fontWeight: "bold" }}>{darwinType}</summary>
        {Object.entries(dict)
          .filter(([k]) => k !== "__darwin_type__")
          .map(([k, v]) => (
            <div key={k} style={{ paddingLeft: 16 }}>
              <span style={{ opacity: 0.7 }}>{k}:</span>{" "}
              <StructuredView data={v} registry={registry} />
            </div>
          ))}
      </details>
    );
  }

  // ── Plain dict — generic ──
  return (
    <details>
      <summary>{"{}"}</summary>
      {Object.entries(dict).map(([k, v]) => (
        <div key={k} style={{ paddingLeft: 16 }}>
          <span style={{ opacity: 0.7 }}>{k}:</span>{" "}
          <StructuredView data={v} registry={registry} />
        </div>
      ))}
    </details>
  );
}

function ListView({
  data,
  registry,
}: {
  data: unknown;
  registry?: DarwinTypeRegistry;
}) {
  const ctxRegistry = useDarwinTypeRegistry();
  registry = registry ?? ctxRegistry;
  if (typeof data !== "object" || data === null) return null;
  const dict = data as Record<string, unknown>;
  const items = (dict["items"] ?? []) as unknown[];
  return (
    <div style={{ paddingLeft: 16 }}>
      {items.map((item, i) => (
        <div key={i} style={{ display: "flex", gap: 8, margin: "1px 0" }}>
          <span style={{ opacity: 0.4 }}>#{i + 1}</span>
          <StructuredView data={item} registry={registry} />
        </div>
      ))}
    </div>
  );
}

// ─── Example: custom renderer for User dataclass ─────────────────────
//
// Register it anywhere before the stream mounts:
//
//   import { darwinTypes } from "./DarwinStream.client";
//   darwinTypes.register("User", UserCard);
//
// When darwin.logjson(user_obj) fires, the stream sees:
//   {"__darwin_type__": "User", "name": "Alice", "age": 30, ...}
// and renders <UserCard /> instead of the generic tree.

interface UserData {
  __darwin_type__: "User";
  name: string;
  age: number;
  email: string;
  addresses?: { __darwin_type__: "Address"; street: string; city: string }[];
  metadata?: Record<string, unknown>;
}

export function UserCard({ data }: { data: Record<string, unknown> }) {
  const user = data as unknown as UserData;
  return (
    <div
      style={{
        border: "1px solid #30363d",
        borderRadius: 6,
        padding: 8,
        margin: "2px 0",
        background: "#161b22",
      }}
    >
      <div style={{ fontWeight: "bold", fontSize: 14, color: "#f0883e" }}>
        {user.name}
      </div>
      <div style={{ fontSize: 12, opacity: 0.7 }}>
        {user.email} · age {user.age}
      </div>
      {user.addresses && user.addresses.length > 0 && (
        <div style={{ marginTop: 4, fontSize: 11, opacity: 0.5 }}>
          {user.addresses.map((a, i) => (
            <div key={i}>
              {a.street}, {a.city}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── React context variant (optional — for prop-drilling avoidance) ──
// Wrap <DarwinStreamProvider registry={myRegistry}> at the root,
// and any StructuredView inside will pick it up automatically.

const RegistryCtx = createContext<DarwinTypeRegistry>(darwinTypes);

export function DarwinStreamProvider({
  registry = darwinTypes,
  children,
}: {
  registry?: DarwinTypeRegistry;
  children: React.ReactNode;
}) {
  return <RegistryCtx.Provider value={registry}>{children}</RegistryCtx.Provider>;
}

function useDarwinTypeRegistry(): DarwinTypeRegistry {
  return useContext(RegistryCtx);
}

function ProgressBar({ id, current, total, label }: {
  id: string;
  current: number;
  total: number;
  label: string;
}) {
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  const barLen = 30;
  const filled = Math.round((barLen * pct) / 100);
  const bar =
    "█".repeat(filled) + "░".repeat(Math.max(0, barLen - filled));
  return (
    <div style={{ fontFamily: "monospace", margin: "2px 0", color: "#58a6ff" }}>
      <span>{label}</span>{" "}
      <span style={{ color: "#3fb950" }}>{bar}</span>{" "}
      <span>
        {current}/{total} ({pct}%)
      </span>
    </div>
  );
}

function SpinnerLine({ id, message }: { id: string; message: string }) {
  const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  const [i, setI] = useState(0);

  useEffect(() => {
    const h = setInterval(() => setI((x) => (x + 1) % frames.length), 80);
    return () => clearInterval(h);
  }, []);

  return (
    <div style={{ fontFamily: "monospace", margin: "2px 0", color: "#888" }}>
      {frames[i]} {message}
    </div>
  );
}

// ─── Main component ─────────────────────────────────────────────────

interface Props {
  url: string;
  maxEvents?: number;
}

export default function DarwinStream({ url, maxEvents = 500 }: Props) {
  const { connected, events, sessions, progress, spinners, clear } =
    useDarwinStream(url);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  const visible = maxEvents ? events.slice(-maxEvents) : events;

  useEffect(() => {
    if (autoScroll) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visible.length, autoScroll]);

  return (
    <div
      style={{
        background: "#0d1117",
        color: "#c9d1d9",
        fontFamily: "monospace",
        fontSize: 13,
        height: "100vh",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* ── Header ── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "8px 12px",
          borderBottom: "1px solid #30363d",
          background: "#161b22",
        }}
      >
        <span
          style={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: connected ? "#3fb950" : "#f85149",
          }}
        />
        <span>{connected ? "Connected" : "Disconnected"}</span>
        <span style={{ opacity: 0.5 }}>|</span>
        <span>{events.length} events</span>
        <span style={{ opacity: 0.5 }}>|</span>
        <span>{sessions.size} sessions</span>
        {progress.size > 0 && (
          <>
            <span style={{ opacity: 0.5 }}>|</span>
            <span style={{ color: "#58a6ff" }}>
              {progress.size} progress bar(s)
            </span>
          </>
        )}
        {spinners.size > 0 && (
          <>
            <span style={{ opacity: 0.5 }}>|</span>
            <span style={{ color: "#d29922" }}>
              {spinners.size} spinner(s)
            </span>
          </>
        )}
        <div style={{ flex: 1 }} />
        <label style={{ fontSize: 11, cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
          />{" "}
          auto-scroll
        </label>
        <button
          onClick={clear}
          style={{
            background: "#21262d",
            border: "1px solid #30363d",
            color: "#c9d1d9",
            cursor: "pointer",
            fontSize: 11,
            padding: "2px 8px",
          }}
        >
          clear
        </button>
      </div>

      {/* ── Active progress bars ── */}
      {progress.size > 0 && (
        <div
          style={{
            padding: "4px 12px",
            borderBottom: "1px solid #30363d",
            background: "#0d1117",
          }}
        >
          {[...progress.entries()].map(([id, p]) => (
            <ProgressBar key={id} id={id} {...p} />
          ))}
        </div>
      )}

      {/* ── Active spinners ── */}
      {spinners.size > 0 && (
        <div
          style={{
            padding: "4px 12px",
            borderBottom: "1px solid #30363d",
            background: "#0d1117",
          }}
        >
          {[...spinners].map((id) => (
            <SpinnerLine key={id} id={id} message="running..." />
          ))}
        </div>
      )}

      {/* ── Event log ── */}
      <div style={{ flex: 1, overflowY: "auto", padding: "4px 12px" }}>
        {visible.map((ev, i) => {
          if (ev.type === "LogEvent") return <LogLine key={i} event={ev} />;
          if (ev.type === "SessionStartEvent")
            return (
              <div
                key={i}
                style={{
                  borderTop: "1px solid #30363d",
                  color: "#58a6ff",
                  fontWeight: "bold",
                  marginTop: 8,
                  paddingTop: 4,
                }}
              >
                Session: {ev.name}
              </div>
            );
          if (ev.type === "SessionEndEvent")
            return (
              <div
                key={i}
                style={{
                  borderBottom: "1px solid #30363d",
                  color: "#8b949e",
                  paddingBottom: 4,
                  marginBottom: 8,
                }}
              >
                [Session ended: {ev.name}]
              </div>
            );
          // Progress/Spinner events are shown live in the header, skip in log
          if (
            ev.type === "ProgressStartEvent" ||
            ev.type === "ProgressUpdateEvent" ||
            ev.type === "ProgressEndEvent" ||
            ev.type === "SpinnerStartEvent" ||
            ev.type === "SpinnerUpdateEvent" ||
            ev.type === "SpinnerEndEvent"
          )
            return null;
          return null;
        })}
        <div ref={bottomRef} />
      </div>

      {/* ── Sessions sidebar (minimal) ── */}
      {sessions.size > 0 && (
        <div
          style={{
            borderTop: "1px solid #30363d",
            padding: "4px 12px",
            fontSize: 11,
            background: "#161b22",
            display: "flex",
            gap: 16,
            flexWrap: "wrap",
          }}
        >
          {[...sessions.entries()].map(([name, s]) => (
            <span key={name} style={{ opacity: 0.6 }}>
              {name}: {s.logCount} logs
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
