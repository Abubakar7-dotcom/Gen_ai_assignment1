import { useEffect, useRef, useState } from "react";
import { getSamples, sampleUrl } from "../api";
import { LABELS } from "./ui";

const CONDITIONS = ["none", "salt_pepper", "blur", "occlusion"];
const SEVERITIES = ["low", "medium", "high", "random"];

/**
 * Input selection shared by the three restoration workspaces.
 * value = { source: {file}|{sample}, preview, condition, severity, seed }
 */
export default function InputCard({ value, onChange, onRun, running, runLabel = "Restore", filter = "pet" }) {
  const [samples, setSamples] = useState([]);
  const fileRef = useRef(null);
  const [drag, setDrag] = useState(false);

  useEffect(() => {
    getSamples().then((r) => setSamples(r.samples.filter((s) => s.startsWith(filter)))).catch(() => setSamples([]));
  }, [filter]);

  const setFile = (file) => {
    if (!file) return;
    onChange({ ...value, source: { file }, preview: URL.createObjectURL(file) });
  };
  const set = (k, v) => onChange({ ...value, [k]: v });

  return (
    <div className="card flex flex-col gap-4">
      <div>
        <div className="label mb-2">Input image</div>
        <div
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); setFile(e.dataTransfer.files?.[0]); }}
          className={`flex cursor-pointer items-center gap-3 rounded-lg border-2 border-dashed p-3 transition ${drag ? "border-accent-500 bg-accent-50 dark:bg-slate-800" : "border-slate-200 dark:border-slate-700"}`}>
          {value.preview
            ? <img src={value.preview} className="h-16 w-16 rounded-md object-cover" alt="selected" />
            : <div className="flex h-16 w-16 items-center justify-center rounded-md bg-slate-100 text-2xl dark:bg-slate-800">⬆</div>}
          <div className="text-sm">
            <div className="font-medium">Drop an image or click to upload</div>
            <div className="text-slate-500">PNG / JPEG, max 10 MB. Resized to 128×128.</div>
          </div>
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => setFile(e.target.files?.[0])} />
        </div>
        {samples.length > 0 && (
          <div className="mt-3">
            <div className="mb-1.5 text-xs text-slate-500">…or pick a clean test sample</div>
            <div className="flex flex-wrap gap-2">
              {samples.map((s) => (
                <button key={s} title={s} onClick={() => onChange({ ...value, source: { sample: s }, preview: sampleUrl(s) })}
                  className={`overflow-hidden rounded-md border-2 ${value.source?.sample === s ? "border-accent-600" : "border-transparent"}`}>
                  <img src={sampleUrl(s)} alt={s} className="h-12 w-12 object-cover" />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div>
        <div className="label mb-2">Apply corruption</div>
        <div className="flex flex-wrap gap-2">
          {CONDITIONS.map((c) => (
            <button key={c} onClick={() => set("condition", c)} className={`chip ${value.condition === c ? "chip-on" : "chip-off"}`}>
              {c === "none" ? "None (already corrupted)" : LABELS[c]}
            </button>
          ))}
        </div>
        {value.condition !== "none" && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {SEVERITIES.map((s) => (
              <button key={s} onClick={() => set("severity", s)} className={`chip capitalize ${value.severity === s ? "chip-on" : "chip-off"}`}>{s}</button>
            ))}
            <label className="ml-auto flex items-center gap-2 text-sm">
              Seed
              <input type="number" value={value.seed} onChange={(e) => set("seed", e.target.value)}
                className="w-24 rounded-lg border border-slate-200 bg-transparent px-2 py-1.5 dark:border-slate-700" />
            </label>
          </div>
        )}
      </div>

      <button className="btn" disabled={!value.source || running} onClick={onRun}>
        {running ? "Running…" : runLabel}
      </button>
    </div>
  );
}

export const defaultInput = { source: null, preview: null, condition: "blur", severity: "medium", seed: 42 };

export function corruptionFields(v) {
  return v.condition === "none" ? {} : { condition: v.condition, severity: v.severity, seed: v.seed };
}
