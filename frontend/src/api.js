// Thin client for the FastAPI backend. All requests go through /api (proxied by Vite in dev, nginx in Docker).
const BASE = import.meta.env.VITE_API_BASE || "/api";

async function handle(res) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}

export const getHealth = () => fetch(`${BASE}/health`).then(handle);
export const getSamples = () => fetch(`${BASE}/samples`).then(handle);
export const sampleUrl = (name) => `${BASE}/samples/${encodeURIComponent(name)}`;

/** POST multipart form. `source` = { file } | { sample }; extra = other form fields. */
export function postForm(path, source, extra = {}) {
  const fd = new FormData();
  if (source?.file) fd.append("file", source.file);
  else if (source?.sample) fd.append("sample", source.sample);
  for (const [k, v] of Object.entries(extra)) if (v !== undefined && v !== null && v !== "") fd.append(k, v);
  return fetch(`${BASE}${path}`, { method: "POST", body: fd }).then(handle);
}
