import { useEffect, useState } from "react";
import { api } from "./api";
import AddStreamBar from "./components/AddStreamBar";
import EventPanel from "./components/EventPanel";
import StreamTile from "./components/StreamTile";
import type { Health } from "./types";
import { useLive } from "./useLive";

export default function App() {
  const { connected, streams, live, events, lastError, upsertStream, dropStream } = useLive();
  const [health, setHealth] = useState<Health | null>(null);
  const [now, setNow] = useState(() => performance.now());

  // re-render 4x/s so box fade-out and alert expiry happen without new messages
  useEffect(() => {
    const t = window.setInterval(() => setNow(performance.now()), 250);
    return () => window.clearInterval(t);
  }, []);

  useEffect(() => {
    const load = () => api.health().then(setHealth).catch(() => setHealth(null));
    load();
    const t = window.setInterval(load, 5000);
    return () => window.clearInterval(t);
  }, []);

  const max = health?.max_streams ?? 4;
  const slots = Array.from({ length: max }, (_, i) => streams[i]);

  return (
    <div className="app">
      <header>
        <h1>realtime-vlm</h1>
        <span className={`dot ${connected ? "ok" : "bad"}`} title="WebSocket" />
        <span className="muted">{connected ? "연결됨" : "연결 끊김 — 재시도 중"}</span>
        <span className="spacer" />
        {health ? (
          <span className="muted">
            <span className={`dot ${health.serving.ok ? "ok" : "bad"}`} /> {health.serving.backend} · {health.serving.model} ·{" "}
            {health.inference.strategy} ({health.inference.num_frames}프레임 / {health.inference.window_sec}초)
          </span>
        ) : (
          <span className="muted">서버 상태 확인 중…</span>
        )}
      </header>

      <main>
        <AddStreamBar count={streams.length} max={max} onAdded={upsertStream} />
        <section className="grid">
          {slots.map((s, i) =>
            s ? (
              <StreamTile
                key={s.id}
                stream={s}
                live={live[s.id]}
                error={lastError[s.id]}
                now={now}
                onChanged={upsertStream}
                onRemoved={dropStream}
              />
            ) : (
              <div key={`empty-${i}`} className="tile empty">
                <span className="muted">빈 슬롯 {i + 1}</span>
              </div>
            ),
          )}
        </section>
      </main>

      <EventPanel events={events} streams={streams} />
    </div>
  );
}
