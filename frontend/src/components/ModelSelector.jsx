import { Check } from "lucide-react";
import { MODELS, MODEL_KEYS } from "../lib/models";

/**
 * The GBM / Jump-Diffusion / Heston toggle.
 *
 * `calibrated` marks the families that already have fitted params in context,
 * so the choice also answers "which of these can I use without re-fitting".
 */
export function ModelSelector({ value, onChange, calibrated = {} }) {
  return (
    <div className="grid grid-cols-3 gap-2">
      {MODEL_KEYS.map((key) => {
        const model = MODELS[key];
        const isActive = value === key;
        const isCalibrated = Boolean(calibrated[key]);

        return (
          <button
            key={key}
            type="button"
            onClick={() => onChange(key)}
            title={model.blurb}
            className={`flex flex-col items-start gap-1 rounded-lg border px-3 py-2.5 text-left transition ${
              isActive
                ? "border-accent bg-accent/10"
                : "border-edge bg-surface-2 hover:border-ink-faint"
            }`}
          >
            <span className="flex w-full items-center justify-between gap-1">
              <span
                className={`text-sm font-medium ${isActive ? "text-accent" : "text-ink"}`}
              >
                {model.label}
              </span>
              {isCalibrated && <Check size={13} className="shrink-0 text-good" />}
            </span>
            <span className="text-[11px] leading-tight text-ink-faint">
              {isCalibrated ? "calibrated params ready" : "manual params"}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export default ModelSelector;
