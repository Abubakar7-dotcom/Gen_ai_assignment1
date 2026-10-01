import { useEffect, useRef, useState } from "react";
import { getSamples, postForm, sampleUrl } from "../api";
import { ErrorBox, ImagePanel, Stat } from "../components/ui";

const STYLES = [
  { id: 1, name: "Style 1", hint: "FS2K style category 1" },
  { id: 2, name: "Style 2", hint: "FS2K style category 2" },
  { id: 3, name: "Style 3", hint: "FS2K style category 3" },
];

export default function SketchPage() {
  const [source, setSource] = useState(null);
  const [preview, setPreview] = useState(null);
  const [style, setStyle] = useState(1);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);
  const [camOn, setCamOn] = useState(false);
  const [samples, setSamples] = useState([]);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => { getSamples().then((r) => setSamples(r.samples.filter((s) => s.startsWith("face")))).catch(() => {}); }, []);
  useEffect(() => () => stopCam(), []);

  const startCam = async () => {
    setErr(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 480 } });
      streamRef.current = s; setCamOn(true);
      setTimeout(() => { if (videoRef.current) videoRef.current.srcObject = s; }, 0);
    } catch (e) { setErr(`Webcam unavailable: ${e.message}. Browsers only allow cameras on localhost or HTTPS.`); }
  };
  function stopCam() { streamRef.current?.getTracks().forEach((t) => t.stop()); streamRef.current = null; setCamOn(false); }

  const capture = () => {
    const v = videoRef.current, size = Math.min(v.videoWidth, v.videoHeight);
    const c = document.createElement("canvas"); c.width = c.height = 256;
    c.getContext("2d").drawImage(v, (v.videoWidth - size) / 2, (v.videoHeight - size) / 2, size, size, 0, 0, 256, 256);
    c.toBlob((b) => { const f = new File([b], "webcam.png", { type: "image/png" }); setSource({ file: f }); setPreview(URL.createObjectURL(f)); stopCam(); }, "image/png");
  };

  const run = async () => {
    setRunning(true); setErr(null);
    try { setRes(await postForm("/sketch", source, { style })); } catch (e) { setErr(e.message); } finally { setRunning(false); }
  };

  const download = () => { const a = document.createElement("a"); a.href = res.sketch; a.download = `sketch_style${res.style}.png`; a.click(); };

  return (
    <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
      <div className="flex flex-col gap-4">
        <div className="card flex flex-col gap-4">
          <div>
            <div className="label mb-2">Facial photograph</div>
            <div className="flex gap-2">
              <button className="btn-ghost flex-1" onClick={() => fileRef.current?.click()}>⬆ Upload photo</button>
              {!camOn ? <button className="btn-ghost flex-1" onClick={startCam}>📷 Use webcam</button>
                      : <button className="btn-ghost flex-1" onClick={stopCam}>Stop webcam</button>}
              <input ref={fileRef} type="file" accept="image/*" hidden
                     onChange={(e) => { const f = e.target.files?.[0]; if (f) { setSource({ file: f }); setPreview(URL.createObjectURL(f)); } }} />
            </div>
            {camOn && (
              <div className="mt-3 flex flex-col gap-2">
                <video ref={videoRef} autoPlay playsInline muted className="aspect-square w-full rounded-lg bg-black object-cover" />
                <button className="btn" onClick={capture}>Capture</button>
              </div>
            )}
            {samples.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {samples.map((s) => (
                  <button key={s} onClick={() => { setSource({ sample: s }); setPreview(sampleUrl(s)); }}
                          className={`overflow-hidden rounded-md border-2 ${source?.sample === s ? "border-accent-600" : "border-transparent"}`}>
                    <img src={sampleUrl(s)} className="h-12 w-12 object-cover" alt={s} />
                  </button>
                ))}
              </div>
            )}
          </div>
          <div>
            <div className="label mb-2">Sketch style</div>
            <div className="grid grid-cols-3 gap-2">
              {STYLES.map((s) => (
                <button key={s.id} onClick={() => setStyle(s.id)}
                        className={`rounded-lg border-2 p-3 text-left transition ${style === s.id ? "border-accent-600 bg-accent-50 dark:bg-slate-800" : "border-slate-200 dark:border-slate-700"}`}>
                  <div className="font-semibold">{s.name}</div>
                  <div className="text-xs text-slate-500">{s.hint}</div>
                </button>
              ))}
            </div>
          </div>
          <button className="btn" disabled={!source || running} onClick={run}>{running ? "Generating…" : "Generate sketch"}</button>
        </div>
        <ErrorBox error={err} />
      </div>

      <div className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-2 sm:gap-4">
          <ImagePanel title="Photo (128×128 model input)" src={res?.photo || preview} />
          <ImagePanel title={`Generated sketch${res ? ` · Style ${res.style}` : ""}`} src={res?.sketch} />
        </div>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
          <Stat label="Style condition" value={res ? `Style ${res.style}` : null} hint="learned style embedding" />
          <Stat label="Inference time" value={res ? `${res.inference_ms} ms` : null} hint="generator only · ONNX CPU" />
          <div className="card flex items-center p-3">
            <button className="btn-ghost w-full" disabled={!res} onClick={download}>⬇ Download sketch</button>
          </div>
        </div>
      </div>
    </div>
  );
}
