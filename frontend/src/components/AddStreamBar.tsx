import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { StreamInfo, VideoFile } from "../types";

type Props = { count: number; max: number; onAdded: (s: StreamInfo) => void };

export default function AddStreamBar({ count, max, onAdded }: Props) {
  const [url, setUrl] = useState("");
  const [prompt, setPrompt] = useState("넘어지는 사람");
  const [name, setName] = useState("");
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ text: string; error: boolean } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const full = count >= max;

  const refreshVideos = () => api.videos().then(setVideos).catch(() => undefined);
  useEffect(() => {
    refreshVideos();
  }, []);

  const upload = async (f: File) => {
    setBusy(true);
    setMsg({ text: `업로드 중: ${f.name}`, error: false });
    try {
      const r = await api.upload(f);
      setUrl(r.path);
      setMsg({ text: `업로드 완료: ${r.path}`, error: false });
      refreshVideos();
    } catch (e) {
      setMsg({ text: `업로드 실패: ${(e as Error).message}`, error: true });
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const add = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setMsg(null);
    try {
      const s = await api.addStream({ url: url.trim(), prompt: prompt.trim(), name: name.trim() || undefined });
      onAdded(s);
      setUrl("");
      setName("");
    } catch (err) {
      setMsg({ text: (err as Error).message, error: true });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="addbar" onSubmit={add}>
      <input
        className="url"
        list="video-list"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        onFocus={refreshVideos}
        placeholder="rtsp://... 또는 videos/xxx.mp4 (목록에서 선택)"
        disabled={full}
      />
      <datalist id="video-list">
        {videos.map((v) => (
          <option key={v.path} value={v.path}>{(v.size / 1e6).toFixed(1)} MB</option>
        ))}
      </datalist>
      <button type="button" disabled={busy || full} onClick={() => fileRef.current?.click()}>MP4 업로드</button>
      <input
        ref={fileRef}
        type="file"
        accept="video/*,.mp4,.mkv,.avi,.mov,.webm,.ts"
        hidden
        onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
      />
      <input className="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="찾을 대상" disabled={full} />
      <input className="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="이름 (선택)" disabled={full} />
      <button type="submit" className="primary" disabled={busy || full || !url.trim() || !prompt.trim()}>
        추가 ({count}/{max})
      </button>
      {msg && <span className={`addbar-msg ${msg.error ? "error" : ""}`}>{msg.text}</span>}
    </form>
  );
}
