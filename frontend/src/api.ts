import type { DetEvent, Health, StreamInfo, StreamSettings, VideoFile } from "./types";

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "content-type": "application/json" };
  }
  const r = await fetch(url, init);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return (r.status === 204 ? undefined : await r.json()) as T;
}

export const api = {
  health: () => req<Health>("GET", "/api/health"),
  streams: () => req<StreamInfo[]>("GET", "/api/streams"),
  addStream: (b: { url: string; prompt: string; name?: string } & Partial<StreamSettings>) =>
    req<StreamInfo>("POST", "/api/streams", b),
  updateStream: (id: string, b: Partial<StreamSettings> & { name?: string }) =>
    req<StreamInfo>("PATCH", `/api/streams/${id}`, b),
  removeStream: (id: string) => req<void>("DELETE", `/api/streams/${id}`),
  videos: () => req<VideoFile[]>("GET", "/api/videos"),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<VideoFile>("POST", "/api/upload", fd);
  },
  events: (limit = 200) => req<DetEvent[]>("GET", `/api/events?limit=${limit}`),
  clearEvents: () => req<{ deleted: number }>("DELETE", "/api/events"),
};

export const mjpegUrl = (id: string) => `/api/streams/${id}/mjpeg`;
export const snapshotUrl = (e: DetEvent) => (e.snapshot ? `/event-files/${e.snapshot}` : "");
