import { useEffect, useRef, useState } from "react";

export const LABELS = { clean: "Clean", salt_pepper: "Salt & pepper", blur: "Gaussian blur", occlusion: "Occlusion" };
export const BRANCH_LABELS = {
  clean: "Identity", salt_pepper: "Salt & pepper expert", blur: "Blur expert", occlusion: "Occlusion expert",
};
export const BRANCH_COLORS = { clean: "#94a3b8", salt_pepper: "#f59e0b", blur: "#6366f1", occlusion: "#10b981" };

export function ImagePanel({ title, src, children, placeholder = "No image yet" }) {
  return (
    <div className="card flex flex-col gap-2 p-3">
      <div className="label">{title}</div>
      <div className="aspect-square w-full overflow-hidden rounded-lg bg-slate-100 dark:bg-slate-800">
        {children ? children : src ? (
          <img src={src} alt={title} className="pixel h-full w-full object-contain" />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-slate-400">{placeholder}</div>
        )}
      </div>
    </div>
  );
}

export function Stat({ label, value, hint }) {
  return (
    <div className="card p-3">
      <div className="label">{label}</div>
      <div className="mt-1 text-lg font-semibold tabular-nums">{value ?? "—"}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

export function Bars({ title, values, labels, highlight, colors }) {
  const entries = Object.entries(values || {});
  return (
    <div className="card">
      <div className="label mb-3">{title}</div>
      {entries.length === 0 && <div className="text-sm text-slate-400">Run the model to see values.</div>}
      <div className="flex flex-col gap-2.5">
        {entries.map(([k, v]) => (
          <div key={k}>
            <div className="mb-1 flex justify-between text-sm">
              <span className={k === highlight ? "font-semibold text-accent-600 dark:text-accent-500" : ""}>{labels[k] || k}</span>
              <span className="tabular-nums">{(v * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2.5 rounded-full bg-slate-100 dark:bg-slate-800">
              <div className="h-2.5 rounded-full transition-all"
                   style={{ width: `${Math.max(v * 100, 0.5)}%`, background: colors?.[k] || (k === highlight ? "#4f46e5" : "#a5b4fc") }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function StackedBar({ values }) {
  if (!values) return null;
  return (
    <div className="flex h-4 w-full overflow-hidden rounded-full">
      {Object.entries(values).map(([k, v]) => (
        <div key={k} title={`${BRANCH_LABELS[k]} ${(v * 100).toFixed(1)}%`} style={{ width: `${v * 100}%`, background: BRANCH_COLORS[k] }} />
      ))}
    </div>
  );
}

/** |output - clean| averaged over RGB, drawn with an inferno-like colour ramp (computed client-side). */
export function ErrorMap({ clean, output }) {
  const ref = useRef(null);
  const [meanErr, setMeanErr] = useState(null);
  useEffect(() => {
    if (!clean || !output) return;
    let cancelled = false;
    const load = (src) => new Promise((ok) => { const i = new Image(); i.onload = () => ok(i); i.src = src; });
    Promise.all([load(clean), load(output)]).then(([a, b]) => {
      if (cancelled) return;
      const w = a.width, h = a.height, cv = ref.current;
      cv.width = w; cv.height = h;
      const ctx = cv.getContext("2d");
      const read = (img) => { ctx.drawImage(img, 0, 0, w, h); return ctx.getImageData(0, 0, w, h).data; };
      const da = read(a), db = read(b), out = ctx.createImageData(w, h);
      let sum = 0;
      for (let p = 0; p < da.length; p += 4) {
        const e = (Math.abs(da[p] - db[p]) + Math.abs(da[p + 1] - db[p + 1]) + Math.abs(da[p + 2] - db[p + 2])) / (3 * 255);
        sum += e;
        const t = Math.min(e / 0.5, 1); // same 0..0.5 scale as the report figures
        out.data[p] = 255 * Math.min(1, 1.6 * t); out.data[p + 1] = 255 * Math.max(0, 1.6 * t - 0.6);
        out.data[p + 2] = 255 * (t < 0.4 ? 0.6 * t / 0.4 : Math.max(0, 0.6 - (t - 0.4))); out.data[p + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
      setMeanErr(sum / (w * h));
    });
    return () => { cancelled = true; };
  }, [clean, output]);
  if (!clean || !output) return <div className="flex h-full items-center justify-center p-4 text-center text-sm text-slate-400">Needs a known clean image (pick a sample + corruption)</div>;
  return (
    <div className="relative h-full w-full">
      <canvas ref={ref} className="h-full w-full object-contain" />
      {meanErr !== null && <span className="absolute bottom-1 right-1 rounded bg-black/60 px-1.5 text-xs text-white">MAE {meanErr.toFixed(3)}</span>}
    </div>
  );
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">{error}</div>;
}

export function describeParams(c) {
  if (!c) return "none (input used as uploaded)";
  const p = c.params || {};
  if (c.condition === "salt_pepper") return `p = ${p.p?.toFixed(3)}`;
  if (c.condition === "blur") return `kernel ${p.kernel}, σ ${Number(p.sigma).toFixed(2)}`;
  if (c.condition === "occlusion") return `${p.n} rect(s), ${(p.area * 100).toFixed(1)}% area`;
  return "clean";
}
