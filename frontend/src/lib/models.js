/**
 * Descriptions of the three model families, mirroring the Pydantic params
 * models in src/calibration/.
 *
 * This is the single place the UI learns which fields a model carries, what
 * to call them, and what a sane manual default is. Anything that needs to
 * render or build a params object reads it from here, so adding a model means
 * touching one file rather than every form.
 *
 * `model_type` is the discriminator the API validates against — it always
 * travels with the params object.
 */

export const MODELS = {
  gbm: {
    key: "gbm",
    label: "GBM",
    fullName: "Geometric Brownian Motion",
    blurb: "Constant drift and volatility. Exact at any step size.",
    fields: [
      { key: "mu", label: "μ", name: "drift", default: 0.08, step: 0.01 },
      { key: "sigma", label: "σ", name: "volatility", default: 0.2, step: 0.01 },
    ],
  },
  jump_diffusion: {
    key: "jump_diffusion",
    label: "Jump-Diffusion",
    fullName: "Merton Jump-Diffusion",
    blurb: "GBM plus compensated log-normal jumps at a Poisson rate.",
    fields: [
      { key: "mu", label: "μ", name: "drift", default: 0.08, step: 0.01 },
      { key: "sigma", label: "σ", name: "diffusion vol", default: 0.2, step: 0.01 },
      { key: "lambda_j", label: "λⱼ", name: "jump intensity", default: 5, step: 0.5 },
      { key: "mu_j", label: "μⱼ", name: "mean jump", default: -0.02, step: 0.01 },
      { key: "sigma_j", label: "σⱼ", name: "jump vol", default: 0.05, step: 0.01 },
    ],
  },
  heston: {
    key: "heston",
    label: "Heston",
    fullName: "Heston Stochastic Volatility",
    blurb: "Mean-reverting variance, simulated with Andersen's QE scheme.",
    fields: [
      { key: "kappa", label: "κ", name: "mean reversion", default: 2, step: 0.1 },
      { key: "theta", label: "θ", name: "long-run variance", default: 0.04, step: 0.01 },
      { key: "xi", label: "ξ", name: "vol of vol", default: 0.3, step: 0.05 },
      { key: "rho", label: "ρ", name: "spot/vol corr", default: -0.6, step: 0.05 },
      { key: "v0", label: "v₀", name: "initial variance", default: 0.04, step: 0.01 },
    ],
    // Fitted by the calibrator, reported but never edited by hand.
    derivedFields: [{ key: "feller_satisfied", label: "Feller", name: "2κθ > ξ²" }],
  },
};

export const MODEL_KEYS = ["gbm", "jump_diffusion", "heston"];

export function defaultParams(modelType) {
  const model = MODELS[modelType];
  const params = { model_type: modelType };
  for (const field of model.fields) params[field.key] = field.default;
  if (modelType === "heston") params.feller_satisfied = true;
  return params;
}

/** Label for a params key, falling back to the raw key for anything unknown. */
export function fieldLabel(modelType, key) {
  const model = MODELS[modelType];
  const all = [...(model?.fields ?? []), ...(model?.derivedFields ?? [])];
  return all.find((f) => f.key === key) ?? { key, label: key, name: "" };
}

/**
 * Option types the backend's PAYOFF_REGISTRY exposes, with the argument rules
 * src/api/schemas.py enforces: lookbacks are floating-strike and take no K,
 * knock-outs require a barrier and nothing else accepts one. The forms read
 * these flags so a request is never built in a shape the API will reject.
 */
export const OPTION_TYPES = [
  { key: "european_call", label: "European Call", needsStrike: true, needsBarrier: false },
  { key: "european_put", label: "European Put", needsStrike: true, needsBarrier: false },
  { key: "asian_call", label: "Asian Call", needsStrike: true, needsBarrier: false },
  { key: "asian_put", label: "Asian Put", needsStrike: true, needsBarrier: false },
  {
    key: "barrier_ko_call",
    label: "Barrier KO Call (up-and-out)",
    needsStrike: true,
    needsBarrier: true,
  },
  {
    key: "barrier_ko_put",
    label: "Barrier KO Put (down-and-out)",
    needsStrike: true,
    needsBarrier: true,
  },
  { key: "lookback_call", label: "Lookback Call (floating strike)", needsStrike: false, needsBarrier: false },
  { key: "lookback_put", label: "Lookback Put (floating strike)", needsStrike: false, needsBarrier: false },
];

export function optionSpec(key) {
  return OPTION_TYPES.find((o) => o.key === key) ?? OPTION_TYPES[0];
}

export const VARIANCE_REDUCTION = [
  { key: "", label: "None" },
  { key: "antithetic", label: "Antithetic" },
  { key: "stratified", label: "Stratified" },
];

export const TRADING_DAYS = 252;

// Rounded rather than 1/252 exactly, so the input box shows 0.003968 instead
// of 0.003968253968253968. The backend derives n_steps = round(T/dt), which
// still lands on 252 steps for a one-year horizon.
export const DEFAULT_DT = Number((1 / TRADING_DAYS).toFixed(6));

export const MAX_SIMULATIONS = 100000;
