import { useEffect, useState } from "react";
import { getHealth } from "./api";
import RestorePage from "./pages/RestorePage";
import SketchPage from "./pages/SketchPage";

const TABS = [
  { id: "universal", name: "Universal Restoration", desc: "Task 1 · one denoising autoencoder for every input condition" },
  { id: "hard", name: "Hard-Routed Restoration", desc: "Task 2 · corruption classifier → specialist autoencoder (clean inputs bypass)" },
  { id: "moe", name: "Soft Mixture-of-Experts Restoration", desc: "Task 3 · gate blends identity + three experts with continuous weights" },
  { id: "sketch", name: "Face-to-Sketch Generator", desc: "Task 4 · style-conditioned pix2pix (U-Net + PatchGAN)" },
];

function HealthPill() {
  const [h, setH] = useState(null);
  const [down, setDown] = useState(false);
  useEffect(() => {
    const poll = () => getHealth().then((r) => { setH(r); setDown(false); }).catch(() => setDown(true));
    poll(); const t = setInterval(poll, 15000); return () => clearInterval(t);
  }, []);
  const models = h ? Object.values(h.models) : [];
  const avail = models.filter((m) => m.available).length;
  const ok = !down && h && avail === models.length;
  return (
    <span title={h ? Object.entries(h.models).map(([k, v]) => `${k}: ${v.available ? "ok" : "missing"}`).join("\n") : ""}
          className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${ok ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300" : "bg-amber-50 text-amber-700 dark:bg-amber-950 dark:text-amber-300"}`}>
      <i className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : "bg-amber-500"}`} />
      {down ? "API offline" : h ? `API online · ${avail}/${models.length} models` : "connecting…"}
    </span>
  );
}

export default function App() {
  const [tab, setTab] = useState(() => location.hash.slice(1) || "universal");
  const [dark, setDark] = useState(() => matchMedia("(prefers-color-scheme: dark)").matches);
  useEffect(() => { document.documentElement.classList.toggle("dark", dark); }, [dark]);
  useEffect(() => { if (location.hash.slice(1) !== tab) location.hash = tab; }, [tab]);
  useEffect(() => {   // deep links / back button: #universal, #hard, #moe, #sketch
    const onHash = () => { const h = location.hash.slice(1); if (TABS.some((t) => t.id === h)) setTab(h); };
    addEventListener("hashchange", onHash); return () => removeEventListener("hashchange", onHash);
  }, []);
  const active = TABS.find((t) => t.id === tab) || TABS[0];

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-5 px-4 py-5 sm:px-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent-600 font-bold text-white">R</div>
          <div>
            <h1 className="text-lg font-bold leading-tight">Restoration Lab</h1>
            <p className="text-xs text-slate-500">Generative AI · Assignment 1</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <HealthPill />
          <button className="btn-ghost py-1" onClick={() => setDark(!dark)} aria-label="toggle theme">{dark ? "☀" : "☾"}</button>
        </div>
      </header>

      <nav className="flex gap-1 overflow-x-auto rounded-xl border border-slate-200 bg-white p-1 dark:border-slate-800 dark:bg-slate-900">
        {TABS.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)}
                  className={`whitespace-nowrap rounded-lg px-4 py-2 text-sm font-medium transition ${tab === t.id ? "bg-accent-600 text-white shadow" : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"}`}>
            {t.name}
          </button>
        ))}
      </nav>
      <p className="-mt-2 text-sm text-slate-500">{active.desc}</p>

      {/* keep each workspace mounted so switching tabs does not lose results */}
      {TABS.map((t) => (
        <section key={t.id} hidden={t.id !== tab}>
          {t.id === "sketch" ? <SketchPage /> : <RestorePage mode={t.id} />}
        </section>
      ))}
    </div>
  );
}
