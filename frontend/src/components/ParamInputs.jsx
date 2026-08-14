import { RotateCcw } from "lucide-react";
import { MODELS } from "../lib/models";
import { Badge, Field, NumberInput } from "./ui";

/**
 * Editable model parameters, the counterpart to the read-only ParamDisplay.
 *
 * The simulate, price and risk pages all need the same thing: start from the
 * calibrated params when there are any, but stay editable so the model can be
 * poked at by hand. `onReset` puts the fitted values back after poking.
 *
 * Heston's `feller_satisfied` is never editable here. It is a *fact about* a
 * parameter set (2κθ > ξ²), not an input, so it is recomputed from whatever
 * the boxes currently hold rather than typed in.
 */
export function ParamInputs({ modelType, params, onChange, onReset, canReset }) {
  const model = MODELS[modelType];

  const update = (key, value) => {
    const next = { ...params, [key]: value };
    if (modelType === "heston") {
      next.feller_satisfied = 2 * next.kappa * next.theta > next.xi ** 2;
    }
    onChange(next);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-ink-faint">{model.blurb}</p>
        {canReset && (
          <button
            type="button"
            onClick={onReset}
            className="inline-flex items-center gap-1 text-[11px] text-accent hover:underline"
          >
            <RotateCcw size={11} />
            reset to calibrated
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-3">
        {model.fields.map((field) => (
          <Field key={field.key} label={`${field.label} · ${field.name}`}>
            <NumberInput
              value={params[field.key] ?? ""}
              step={field.step}
              onChange={(value) => update(field.key, value)}
            />
          </Field>
        ))}
      </div>

      {modelType === "heston" && (
        <div className="flex items-center gap-2">
          <Badge tone={params.feller_satisfied ? "good" : "warn"}>
            {params.feller_satisfied ? "Feller satisfied" : "Feller violated"}
          </Badge>
          <span className="tnum text-[11px] text-ink-faint">
            2κθ = {(2 * params.kappa * params.theta).toFixed(4)} vs ξ² ={" "}
            {(params.xi ** 2).toFixed(4)}
          </span>
        </div>
      )}
    </div>
  );
}

export default ParamInputs;
