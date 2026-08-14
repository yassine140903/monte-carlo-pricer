import { fmt } from "../lib/format";
import { MODELS, fieldLabel } from "../lib/models";
import { Badge } from "./ui";

/**
 * Read-only view of one model's calibrated parameters.
 *
 * `model_type` is skipped: it is the discriminator tag the API validates
 * against, not a fitted quantity, and the card is already headed by the model
 * name. Heston's `feller_satisfied` is not a parameter either but does say
 * something about the fit, so it is rendered as a badge instead of a row.
 */
export function ParamDisplay({ modelType, params, className = "" }) {
  if (!params) return null;

  const model = MODELS[modelType];
  const rows = model.fields.filter((field) => params[field.key] !== undefined);
  const feller = params.feller_satisfied;

  return (
    <div className={className}>
      <dl className="divide-y divide-edge/70">
        {rows.map((field) => (
          <div key={field.key} className="flex items-baseline justify-between gap-3 py-1.5">
            <dt className="flex items-baseline gap-2 text-xs">
              <span className="w-6 font-semibold text-accent">{field.label}</span>
              <span className="text-ink-faint">{field.name}</span>
            </dt>
            <dd className="tnum text-sm font-medium text-ink">{fmt(params[field.key])}</dd>
          </div>
        ))}
      </dl>

      {modelType === "heston" && feller !== undefined && (
        <div className="mt-3 flex items-center justify-between gap-2">
          <span className="text-[11px] text-ink-faint">
            {fieldLabel("heston", "feller_satisfied").name}
          </span>
          <Badge tone={feller ? "good" : "warn"}>
            {feller ? "Feller satisfied" : "Feller violated"}
          </Badge>
        </div>
      )}

      {modelType === "heston" && feller === false && (
        <p className="mt-1.5 text-[11px] leading-relaxed text-ink-faint">
          Variance can touch zero. The QE simulation scheme handles this — the
          paths stay valid, the variance process is just more extreme.
        </p>
      )}
    </div>
  );
}

export default ParamDisplay;
