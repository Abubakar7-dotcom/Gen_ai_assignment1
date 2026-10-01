import { useState } from "react";
import { postForm } from "../api";
import InputCard, { corruptionFields, defaultInput } from "../components/InputCard";
import {
  BRANCH_COLORS, BRANCH_LABELS, Bars, ErrorBox, ErrorMap, ImagePanel, LABELS, StackedBar, Stat, describeParams,
} from "../components/ui";

function download(dataUrl, name) {
  const a = document.createElement("a");
  a.href = dataUrl; a.download = name; a.click();
}

/** One component for the three restoration workspaces; `mode` decides endpoint and the extra result cards. */
export default function RestorePage({ mode }) {
  const endpoint = { universal: "/restore/universal", hard: "/restore/hard", moe: "/restore/moe" }[mode];
  const [input, setInput] = useState(defaultInput);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const run = async () => {
    setRunning(true); setErr(null);
    try { setRes(await postForm(endpoint, input.source, corruptionFields(input))); }
    catch (e) { setErr(e.message); }
    finally { setRunning(false); }
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
      <div className="flex flex-col gap-4">
        <InputCard value={input} onChange={setInput} onRun={run} running={running} />
        <ErrorBox error={err} />
      </div>

      <div className="flex flex-col gap-4">
        <div className="grid grid-cols-3 gap-2 sm:gap-4">
          <ImagePanel title="Model input" src={res?.input || input.preview} />
          <ImagePanel title="Restored output" src={res?.output} />
          <ImagePanel title="Error map |output − clean|">
            <ErrorMap clean={res?.corruption?.clean} output={res?.output} />
          </ImagePanel>
        </div>

        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <Stat label="Corruption" value={res ? (res.corruption ? LABELS[res.corruption.condition] : "None") : null}
                hint={res ? describeParams(res.corruption) : null} />
          <Stat label="Severity" value={res?.corruption?.severity ?? (res ? "—" : null)}
                hint={res?.corruption?.seed != null ? `seed ${res.corruption.seed}` : null} />
          <Stat label="Inference time" value={res ? `${res.inference_ms} ms` : null}
                hint={mode === "hard" && res ? `classifier ${res.classifier_ms} ms` : "ONNX Runtime · CPU"} />
          <div className="card flex items-center p-3">
            <button className="btn-ghost w-full" disabled={!res} onClick={() => download(res.output, `restored_${mode}.png`)}>
              ⬇ Download result
            </button>
          </div>
        </div>

        {mode === "hard" && (
          <div className="grid gap-4 md:grid-cols-[1fr_280px]">
            <Bars title="Classifier probabilities" values={res?.probabilities} labels={LABELS} highlight={res?.predicted} />
            <div className="card flex flex-col gap-3">
              <div className="label">Routing decision</div>
              <div>
                <div className="text-xs text-slate-500">Predicted corruption</div>
                <div className="text-lg font-semibold">{res ? LABELS[res.predicted] : "—"}</div>
              </div>
              <div>
                <div className="text-xs text-slate-500">Selected expert</div>
                <div className="inline-block rounded-md bg-accent-100 px-2 py-1 text-sm font-semibold text-accent-700 dark:bg-slate-800 dark:text-accent-500">
                  {res?.expert ?? "—"}
                </div>
              </div>
              {res?.corruption && res.corruption.condition !== res.predicted && (
                <div className="rounded-md bg-amber-50 p-2 text-xs text-amber-800 dark:bg-amber-950 dark:text-amber-300">
                  Misrouted: true corruption was {LABELS[res.corruption.condition]}.
                </div>
              )}
            </div>
          </div>
        )}

        {mode === "moe" && (
          <div className="grid gap-4 md:grid-cols-[1fr_280px]">
            <Bars title="Routing weights (softmax(G(x)/τ))" values={res?.weights} labels={BRANCH_LABELS}
                  highlight={res && Object.entries(res.weights).sort((a, b) => b[1] - a[1])[0][0]} colors={BRANCH_COLORS} />
            <div className="card flex flex-col gap-3">
              <div className="label">Expert mix</div>
              <StackedBar values={res?.weights} />
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
                {Object.entries(BRANCH_LABELS).map(([k, l]) => (
                  <span key={k} className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: BRANCH_COLORS[k] }} />{l}</span>
                ))}
              </div>
              <div>
                <div className="text-xs text-slate-500">Top contributor</div>
                <div className="text-lg font-semibold">{res?.top_expert ?? "—"}</div>
              </div>
              {res && <div className="text-xs text-slate-500">Ranking: {res.ranking.join(" › ")}</div>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
