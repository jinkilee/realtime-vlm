import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { DetEvent, Live, StreamInfo } from "./types";

const MAX_EVENTS = 500;

/** WebSocket /ws subscription: stream status (1 Hz), judge/result per stream, detection events. */
export function useLive() {
  const [connected, setConnected] = useState(false);
  const [streams, setStreams] = useState<StreamInfo[]>([]);
  const [live, setLive] = useState<Record<string, Live>>({});
  const [events, setEvents] = useState<DetEvent[]>([]);
  const [lastError, setLastError] = useState<Record<string, string>>({});
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let closed = false;
    let retry = 500;
    let timer: number | undefined;

    const loadEvents = () => api.events(MAX_EVENTS).then(setEvents).catch(() => undefined);

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);
      wsRef.current = ws;
      ws.onopen = () => {
        setConnected(true);
        retry = 500;
        loadEvents();
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) timer = window.setTimeout(connect, (retry = Math.min(retry * 2, 5000)));
      };
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        const now = performance.now();
        switch (m.type) {
          case "status":
            setStreams(m.streams);
            setLive((prev) => {
              const ids = new Set((m.streams as StreamInfo[]).map((s) => s.id));
              const next: Record<string, Live> = {};
              for (const id of Object.keys(prev)) if (ids.has(id)) next[id] = prev[id];
              return next;
            });
            break;
          case "judge":
            setLive((p) => ({
              ...p,
              [m.stream_id]: {
                ...p[m.stream_id],
                judge: { frame_ts: m.frame_ts, score: m.score, detected: m.detected, latency_ms: m.latency_ms, recv: now },
              },
            }));
            break;
          case "result":
            setLive((p) => ({
              ...p,
              [m.stream_id]: {
                ...p[m.stream_id],
                result: {
                  frame_ts: m.frame_ts, boxes: m.boxes, score: m.score, detected: m.detected,
                  latency_ms: m.latency_ms, recv: now,
                },
              },
            }));
            setLastError((p) => (p[m.stream_id] ? { ...p, [m.stream_id]: "" } : p));
            break;
          case "error":
            setLastError((p) => ({ ...p, [m.stream_id]: m.error }));
            break;
          case "event":
            setEvents((prev) => {
              const e = m.event as DetEvent;
              const rest = prev.filter((x) => x.id !== e.id);
              return [e, ...rest].slice(0, MAX_EVENTS);
            });
            break;
          case "events_cleared":
            setEvents([]);
            break;
        }
      };
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      wsRef.current?.close();
    };
  }, []);

  /** Optimistic local update after REST calls (the next 1 Hz status will confirm). */
  const upsertStream = (s: StreamInfo) =>
    setStreams((prev) => (prev.some((x) => x.id === s.id) ? prev.map((x) => (x.id === s.id ? s : x)) : [...prev, s]));
  const dropStream = (id: string) => setStreams((prev) => prev.filter((x) => x.id !== id));

  return { connected, streams, live, events, lastError, upsertStream, dropStream };
}
