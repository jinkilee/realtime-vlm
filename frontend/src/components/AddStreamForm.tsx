import { useEffect, useId, useRef, useState } from "react";
import { api } from "../api";
import type { StreamInfo, VideoFile } from "../types";

const PROMPT_KEY = "rvlm.lastPrompt";
const DEFAULT_PROMPT = "넘어지는 사람";
const loadPrompt = () => localStorage.getItem(PROMPT_KEY) || DEFAULT_PROMPT;
const savePrompt = (p: string) => localStorage.setItem(PROMPT_KEY, p);

type Msg = { text: string; error: boolean } | null;

type Props = {
  variant: "bar" | "slot";
  count: number;
  max: number;
  onAdded: (s: StreamInfo) => void;
  onDone?: () => void; // slot: close the form after a successful add / cancel
};

/**
 * Add streams by URL/path or by uploading files.
 * Uploading N files adds each as a stream (with the current prompt) while slots remain;
 * the rest are only uploaded (selectable from the list later).
 */
export default function AddStreamForm({ variant, count, max, onAdded, onDone }: Props) {
  const [url, setUrl] = useState("");
  const [prompt, setPrompt] = useState(loadPrompt);
  const [name, setName] = useState("");
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<Msg>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const listId = useId();
  const full = count >= max;

  const refreshVideos = () => api.videos().then(setVideos).catch(() => undefined);
  useEffect(() => {
    refreshVideos();
  }, []);

  const addOne = async (u: string, n?: string) => {
    const s = await api.addStream({ url: u, prompt: prompt.trim(), name: n || undefined });
    onAdded(s);
    savePrompt(prompt.trim());
    return s;
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setMsg(null);
    try {
      await addOne(url.trim(), name.trim());
      setUrl("");
      setName("");
      onDone?.();
    } catch (err) {
      setMsg({ text: (err as Error).message, error: true });
    } finally {
      setBusy(false);
    }
  };

  const uploadFiles = async (files: File[]) => {
    if (!files.length) return;
    setBusy(true);
    let free = max - count;
    const canAdd = prompt.trim().length > 0;
    const added: string[] = [];
    const uploadedOnly: string[] = [];
    const failed: string[] = [];
    for (const [i, f] of files.entries()) {
      setMsg({ text: `업로드 중 (${i + 1}/${files.length}): ${f.name}`, error: false });
      try {
        const r = await api.upload(f);
        if (canAdd && free > 0) {
          await addOne(r.path, f.name);
          added.push(f.name);
          free -= 1;
        } else {
          uploadedOnly.push(r.path);
        }
      } catch (err) {
        failed.push(`${f.name} (${(err as Error).message})`);
      }
    }
    refreshVideos();
    if (fileRef.current) fileRef.current.value = "";
    setBusy(false);

    const parts: string[] = [];
    if (added.length) parts.push(`${added.length}개 추가됨`);
    if (uploadedOnly.length) {
      parts.push(
        canAdd
          ? `${uploadedOnly.length}개는 빈 슬롯이 없어 업로드만 됨 (목록에서 선택 가능)`
          : `${uploadedOnly.length}개 업로드만 됨 (프롬프트를 입력해야 자동 추가)`,
      );
      if (uploadedOnly.length === 1 && !added.length) setUrl(uploadedOnly[0]);
    }
    if (failed.length) parts.push(`실패: ${failed.join(", ")}`);
    setMsg({ text: parts.join(" · "), error: failed.length > 0 });
    if (variant === "slot" && added.length && !failed.length) onDone?.();
  };

  const fileInput = (
    <input
      ref={fileRef}
      type="file"
      multiple
      accept="video/*,.mp4,.mkv,.avi,.mov,.webm,.ts"
      hidden
      onChange={(e) => uploadFiles(Array.from(e.target.files ?? []))}
    />
  );
  const datalist = (
    <datalist id={listId}>
      {videos.map((v) => (
        <option key={v.path} value={v.path}>{(v.size / 1e6).toFixed(1)} MB</option>
      ))}
    </datalist>
  );
  const remaining = Math.max(0, max - count);
  const uploadLabel = variant === "bar" ? "MP4 업로드 (여러 개)" : "MP4 업로드";

  return (
    <form className={variant === "bar" ? "addbar" : "slot-form"} onSubmit={submit} onClick={(e) => e.stopPropagation()}>
      {variant === "slot" && <div className="slot-form-title">스트림 추가 <span className="muted">빈 슬롯 {remaining}개</span></div>}
      <input
        className="url"
        list={listId}
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        onFocus={refreshVideos}
        placeholder="rtsp://... 또는 videos/xxx.mp4 (목록에서 선택)"
        disabled={full}
        autoFocus={variant === "slot"}
      />
      {datalist}
      <input className="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="찾을 대상" disabled={full} />
      {variant === "bar" && (
        <input className="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="이름 (선택)" disabled={full} />
      )}
      <div className="form-buttons">
        <button type="button" disabled={busy} onClick={() => fileRef.current?.click()} title="여러 파일 선택 가능. 빈 슬롯 수만큼 바로 추가됩니다.">
          {uploadLabel}
        </button>
        <button type="submit" className="primary" disabled={busy || full || !url.trim() || !prompt.trim()}>
          {variant === "bar" ? `추가 (${count}/${max})` : "추가"}
        </button>
        {variant === "slot" && (
          <button type="button" onClick={() => onDone?.()} disabled={busy}>취소</button>
        )}
      </div>
      {fileInput}
      {msg && <span className={`form-msg ${msg.error ? "error" : ""}`}>{msg.text}</span>}
    </form>
  );
}
