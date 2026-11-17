import { useEffect, useState } from "react";
import { api, mjpegUrl } from "../api";
import type { Live, StreamInfo } from "../types";

const BOX_TTL_MS = 3000;    // boxes fade out and disappear after this (no newer result)
const ALERT_TTL_MS = 2500;  // tile stays in "detected" state this long after the last positive

type Props = {
  stream: StreamInfo;
  live?: Live;
  error?: string;
  now: number;
  onChanged: (s: StreamInfo) => void;
  onRemoved: (id: string) => void;
};

const STATE_LABEL: Record<string, string> = {
  connecting: "연결 중",
  running: "재생 중",
  reconnecting: "재연결 중",
  error: "오류",
  stopped: "중지",
};

export default function StreamTile({ stream, live, error, now, onChanged, onRemoved }: Props) {
  const { status, settings } = stream;
  const [prompt, setPrompt] = useState(settings.prompt);
  const [editing, setEditing] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    if (!editing) setPrompt(settings.prompt);
  }, [settings.prompt, editing]);

  const w = natural?.w || status.width || 16;
  const h = natural?.h || status.height || 9;

  const result = live?.result;
  const judge = live?.judge;
  const resultAge = result ? now - result.recv : Infinity;
  const boxes = result && resultAge < BOX_TTL_MS ? result.boxes : [];
  const boxOpacity = Math.max(0.25, 1 - resultAge / BOX_TTL_MS);
  const lastPositive = Math.max(
    judge?.detected ? judge.recv : -Infinity,
    result?.detected ? result.recv : -Infinity,
  );
  const alert = now - lastPositive < ALERT_TTL_MS;
  const score = judge && (!result || judge.recv >= result.recv) ? judge.score : result?.score ?? null;

  const patch = async (b: Parameters<typeof api.updateStream>[1]) => {
    setBusy(true);
    setMsg("");
    try {
      onChanged(await api.updateStream(stream.id, b));
      return true;
    } catch (e) {
      setMsg((e as Error).message);
      return false;
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    try {
      await api.removeStream(stream.id);
      onRemoved(stream.id);
    } catch (e) {
      setMsg((e as Error).message);
      setBusy(false);
    }
  };

  const shownError = error || status.error;

  return (
    <div className={`tile ${alert ? "alert" : ""}`}>
      <div className="tile-head">
        <span className="tile-name" title={stream.url}>{stream.name}</span>
        <span className={`badge state-${status.state}`}>{STATE_LABEL[status.state] ?? status.state}</span>
        {alert && <span className="badge detect">감지 {score !== null ? score.toFixed(2) : ""}</span>}
        <span className="spacer" />
        <button className="icon" title="설정" onClick={() => setShowSettings((v) => !v)}>⚙</button>
        <button className="icon danger" title="삭제" disabled={busy} onClick={remove}>✕</button>
      </div>

      <div className="tile-media">
        <div className="frame" style={{ aspectRatio: `${w} / ${h}`, ["--ar" as string]: `${w / h}` }}>
          <img
            src={mjpegUrl(stream.id)}
            alt={stream.name}
            onLoad={(e) => {
              const img = e.currentTarget;
              if (img.naturalWidth && (img.naturalWidth !== natural?.w || img.naturalHeight !== natural?.h))
                setNatural({ w: img.naturalWidth, h: img.naturalHeight });
            }}
          />
          {boxes.map((b, i) => (
            <div
              key={i}
              className="box"
              style={{
                left: `${b.x1 * 100}%`,
                top: `${b.y1 * 100}%`,
                width: `${(b.x2 - b.x1) * 100}%`,
                height: `${(b.y2 - b.y1) * 100}%`,
                opacity: boxOpacity,
              }}
            >
              <span className="box-label">{b.label || settings.prompt}</span>
            </div>
          ))}
        </div>
        {status.state !== "running" && (
          <div className="media-overlay">{STATE_LABEL[status.state] ?? status.state}…</div>
        )}
      </div>

      <div className="tile-stats">
        <span title="원본 디코딩 FPS">src {status.src_fps.toFixed(0)}fps</span>
        <span title="초당 추론 횟수">infer {status.infer_fps.toFixed(1)}/s</span>
        <span title="마지막 추론 지연 (판단+bbox)">{result ? `${result.latency_ms}ms` : "-"}</span>
        <span title="판단 점수 P(yes)">score {score !== null ? score.toFixed(2) : "-"}</span>
        {status.width > 0 && <span>{status.width}×{status.height}</span>}
      </div>

      <form
        className="tile-prompt"
        onSubmit={async (e) => {
          e.preventDefault();
          if (prompt.trim() && prompt !== settings.prompt && (await patch({ prompt: prompt.trim() }))) setEditing(false);
        }}
      >
        <input
          value={prompt}
          onFocus={() => setEditing(true)}
          onBlur={() => setEditing(false)}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="찾을 대상 (예: 넘어지는 사람)"
        />
        <button type="submit" disabled={busy || !prompt.trim() || prompt === settings.prompt}>적용</button>
      </form>

      {showSettings && (
        <SettingsForm settings={settings} busy={busy} onApply={(b) => patch(b).then((ok) => ok && setShowSettings(false))} />
      )}
      {(msg || shownError) && <div className="tile-error">{msg || shownError}</div>}
    </div>
  );
}

function SettingsForm({
  settings,
  busy,
  onApply,
}: {
  settings: StreamInfo["settings"];
  busy: boolean;
  onApply: (b: Partial<StreamInfo["settings"]>) => void;
}) {
  const [v, setV] = useState({
    num_frames: settings.num_frames,
    window_sec: settings.window_sec,
    interval_ms: settings.interval_ms,
    judge_threshold: settings.judge_threshold,
  });
  const num = (k: keyof typeof v) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setV({ ...v, [k]: Number(e.target.value) });
  return (
    <form
      className="tile-settings"
      onSubmit={(e) => {
        e.preventDefault();
        onApply(v);
      }}
    >
      <label>프레임 수<input type="number" min={1} max={4} value={v.num_frames} onChange={num("num_frames")} /></label>
      <label>구간(초)<input type="number" min={0.5} max={4} step={0.5} value={v.window_sec} onChange={num("window_sec")} /></label>
      <label>요청 간격(ms)<input type="number" min={0} step={250} value={v.interval_ms} onChange={num("interval_ms")} /></label>
      <label>판단 임계값<input type="number" min={0} max={1} step={0.05} value={v.judge_threshold} onChange={num("judge_threshold")} /></label>
      <button type="submit" disabled={busy}>저장</button>
    </form>
  );
}
