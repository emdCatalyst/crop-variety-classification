import { Check, Loader2, X } from "lucide-react";

const STAGES = ["queued", "loading", "inferring", "rendering", "done"] as const;

type StepState = "done" | "current" | "pending" | "failed";

function classifyStep(idx: number, currentIdx: number, isFailed: boolean): StepState {
  if (isFailed && idx === currentIdx) return "failed";
  if (idx < currentIdx) return "done";
  if (idx === currentIdx) return "current";
  return "pending";
}

export default function Timeline({
  stage,
  labels,
  failedMessage,
}: {
  stage: string;
  labels: Record<(typeof STAGES)[number], string>;
  failedMessage?: string | null;
}) {
  const isFailed = stage === "failed";
  // Map the coarse "processing" row-status to "loading" and unknown values to
  // "queued" so the first step visually reads as in-progress until a finer
  // event arrives.
  const norm: (typeof STAGES)[number] =
    stage === "processing"
      ? "loading"
      : stage === "failed"
      ? "rendering"
      : (STAGES as readonly string[]).includes(stage)
      ? (stage as (typeof STAGES)[number])
      : "queued";
  const currentIdx = STAGES.indexOf(norm);

  return (
    <ol className="relative">
      {STAGES.map((s, i) => {
        const state = classifyStep(i, currentIdx, isFailed);
        const isLast = i === STAGES.length - 1;
        return (
          <li key={s} className="relative flex gap-3 pb-6 last:pb-0">
            {!isLast && (
              <span
                className={`absolute start-[15px] top-8 w-[2px] h-[calc(100%-1.5rem)] -translate-x-1/2 rtl:translate-x-1/2 ${
                  state === "done" || state === "current"
                    ? "bg-brand-300"
                    : "bg-slate-200"
                }`}
                aria-hidden
              />
            )}
            <StepDot state={state} />
            <div className="min-w-0 pt-0.5">
              <div
                className={`text-sm font-medium ${
                  state === "current"
                    ? "text-brand-700"
                    : state === "done"
                    ? "text-slate-700"
                    : state === "failed"
                    ? "text-red-700"
                    : "text-slate-400"
                }`}
              >
                {labels[s]}
              </div>
              {state === "failed" && failedMessage && (
                <p className="text-xs text-red-600 mt-1 break-words">
                  {failedMessage}
                </p>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function StepDot({ state }: { state: StepState }) {
  const base =
    "shrink-0 w-8 h-8 rounded-full flex items-center justify-center border-2 transition-colors";
  if (state === "done") {
    return (
      <div className={`${base} bg-brand-600 border-brand-600 text-white`}>
        <Check size={16} aria-hidden />
      </div>
    );
  }
  if (state === "current") {
    return (
      <div className={`${base} bg-brand-50 border-brand-500 text-brand-600`}>
        <Loader2 size={16} className="animate-spin" aria-hidden />
      </div>
    );
  }
  if (state === "failed") {
    return (
      <div className={`${base} bg-red-50 border-red-500 text-red-600`}>
        <X size={16} aria-hidden />
      </div>
    );
  }
  return (
    <div className={`${base} bg-white border-slate-300 text-slate-300`}>
      <span className="w-2 h-2 rounded-full bg-slate-300" />
    </div>
  );
}
