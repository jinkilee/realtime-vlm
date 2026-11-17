import { useEffect, useState } from "react";
import { api, snapshotUrl } from "../api";
import type { DetEvent, StreamInfo } from "../types";

const fmtTime = (ts: number) =>
  new Date(ts * 1000).toLocaleTimeString("ko-KR", { hour12: false });
const fmtDate = (ts: number) =>
  new Date(ts * 1000).toLocaleString("ko-KR", { hour12: false });
const NEW_MS = 4000;

type Props = { events: DetEvent[]; streams: StreamInfo[] };

export default function EventPanel({ events, streams }: Props) {
  const [filter, setFilter] = useState("");
  const [open, setOpen] = useState<DetEvent | null>(null);
  const [seenAt] = useState(() => new Map<string, number>());

  const shown = filter ? events.filter((e) => e.stream_id === filter) : events;
  // stream names for the filter include streams that no longer exist but still have events
  const names = new Map<string, string>();
  for (const e of events) names.set(e.stream_id, e.stream_name);
  for (const s of streams) names.set(s.id, s.name);

  const clear = async () => {
    if (!confirm("이벤트 로그와 스냅샷을 모두 삭제할까요?")) return;
    await api.clearEvents().catch((e) => alert((e as Error).message));
  };

  return (
    <aside className="events">
      <div className="events-head">
        <h2>이벤트 로그 <span className="muted">{shown.length}</span></h2>
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="">전체 스트림</option>
          {[...names].map(([id, n]) => (
            <option key={id} value={id}>{n}</option>
          ))}
        </select>
        <button className="danger" onClick={clear} disabled={!events.length}>지우기</button>
      </div>
      <ul className="events-list">
        {shown.length === 0 && <li className="muted empty">아직 감지된 이벤트가 없습니다.</li>}
        {shown.map((e) => {
          const key = `${e.id}:${e.count}`;
          if (!seenAt.has(key)) seenAt.set(key, Date.now());
          const fresh = Date.now() - (seenAt.get(key) ?? 0) < NEW_MS;
          return (
            <li key={e.id} className={`event ${fresh ? "fresh" : ""}`} onClick={() => setOpen(e)}>
              {e.snapshot ? <img src={snapshotUrl(e)} alt="" loading="lazy" /> : <div className="noimg">no image</div>}
              <div className="event-body">
                <div className="event-top">
                  <b>{fmtTime(e.ts_start)}</b>
                  <span className="muted">{e.stream_name}</span>
                </div>
                <div className="event-prompt">{e.prompt}</div>
                <div className="muted small">
                  {e.count}회 · {Math.max(0, e.last_seen - e.ts_start).toFixed(1)}초
                  {e.max_score !== null && ` · 최고 ${e.max_score.toFixed(2)}`}
                </div>
              </div>
            </li>
          );
        })}
      </ul>
      {open && <EventModal e={events.find((x) => x.id === open.id) ?? open} onClose={() => setOpen(null)} />}
    </aside>
  );
}

function EventModal({ e, onClose }: { e: DetEvent; onClose: () => void }) {
  useEffect(() => {
    const onKey = (k: KeyboardEvent) => k.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={(x) => x.stopPropagation()}>
        <button className="icon modal-close" onClick={onClose}>✕</button>
        {e.snapshot ? <img src={snapshotUrl(e)} alt="" /> : <div className="noimg big">스냅샷 없음</div>}
        <table>
          <tbody>
            <tr><th>스트림</th><td>{e.stream_name}</td></tr>
            <tr><th>프롬프트</th><td>{e.prompt}</td></tr>
            <tr><th>시작</th><td>{fmtDate(e.ts_start)}</td></tr>
            <tr><th>마지막 감지</th><td>{fmtDate(e.last_seen)}</td></tr>
            <tr><th>감지 횟수</th><td>{e.count}회</td></tr>
            <tr><th>최고 점수</th><td>{e.max_score !== null ? e.max_score.toFixed(3) : "-"}</td></tr>
            <tr><th>박스</th><td>{e.boxes.map((b) => `[${[b.x1, b.y1, b.x2, b.y2].map((v) => v.toFixed(2)).join(", ")}]`).join(" ") || "-"}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
