import { useEffect, useMemo, useState } from "react";
import { Play, Sigma } from "lucide-react";
import ConfidenceCone from "../components/ConfidenceCone";
import GreeksTable from "../components/GreeksTable";
import ModelSelector from "../components/ModelSelector";
import ParamInputs from "../components/ParamInputs";
import PayoffHistogram from "../components/PayoffHistogram";
import { LoadingPanel, LoadingSpinner } from "../components/LoadingSpinner";
import {
  Button,
  Card,
  EmptyState,
  ErrorBanner,
  Field,
  NumberInput,
  Select,
  Stat,
} from "../components/ui";
import { useApp } from "../context/AppContext";
import useModelParams from "../hooks/useModelParams";
import usePricing from "../hooks/usePricing";
import useSimulation from "../hooks/useSimulation";
import { fmtMillis, fmtMoney, fmtPercent } from "../lib/format";
import {
  DEFAULT_DT,
  MAX_SIMULATIONS,
  OPTION_TYPES,
  VARIANCE_REDUCTION,
  optionSpec,
} from "../lib/models";

export function SimulatePricePage() {
  const { calibratedParams, selectedTicker, S0, setS0 } = useApp();

  // Model, params, horizon and path count are shared by both sections: the
  // price is meant to be the option written on the process just simulated.
  const [modelType, setModelType] = useState("gbm");
  const model = useModelParams(modelType);

  const [T, setT] = useState(1);
  const [dt, setDt] = useState(DEFAULT_DT);
  const [nSimulations, setNSimulations] = useState(10000);
  const [varianceReduction, setVarianceReduction] = useState("");
  const [seed, setSeed] = useState(42);

  const [optionType, setOptionType] = useState("european_call");
  const [K, setK] = useState(100);
  const [r, setR] = useState(0.05);
  const [q, setQ] = useState(0);
  const [barrier, setBarrier] = useState(120);

  const simulation = useSimulation();
  const pricing = usePricing();

  const spec = optionSpec(optionType);

  // A newly calibrated ticker brings a new spot with it; the strike follows so
  // the default option is at the money rather than wildly off it.
  useEffect(() => {
    setK(Number(S0.toFixed(2)));
    setBarrier(Number((S0 * 1.2).toFixed(2)));
  }, [S0]);

  const tooManyPaths = nSimulations > MAX_SIMULATIONS;

  const commonBody = useMemo(
    () => ({
      S0: Number(S0),
      model_params: model.body,
      T: Number(T),
      dt: Number(dt),
      n_simulations: Number(nSimulations),
      seed: seed === "" ? null : Number(seed),
      variance_reduction: varianceReduction || null,
    }),
    [S0, model.body, T, dt, nSimulations, seed, varianceReduction],
  );

  const onSimulate = () => simulation.run(commonBody);

  const onPrice = () =>
    pricing.run({
      ...commonBody,
      option_type: optionType,
      // The API rejects a strike on a floating-strike lookback and a barrier
      // on anything that is not a knock-out, so these are omitted rather than
      // sent as nulls the schema would have to forgive.
      K: spec.needsStrike ? Number(K) : null,
      r: Number(r),
      q: Number(q),
      barrier: spec.needsBarrier ? Number(barrier) : null,
    });

  const summary = simulation.data?.summary;
  const priced = pricing.data;

  return (
    <div className="space-y-5">
      {/* ---------------------------------------------------------------- */}
      <Card
        title="Simulate"
        subtitle={
          selectedTicker
            ? `Physical measure — the calibrated drift for ${selectedTicker}, not r − q.`
            : "Physical measure — the calibrated drift, not r − q."
        }
        right={
          <Button onClick={onSimulate} disabled={simulation.loading || tooManyPaths}>
            {simulation.loading ? (
              <LoadingSpinner label="Simulating…" />
            ) : (
              <>
                <Play size={15} /> Simulate
              </>
            )}
          </Button>
        }
      >
        <div className="grid gap-5 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
          <div className="space-y-4">
            <ModelSelector
              value={modelType}
              onChange={setModelType}
              calibrated={calibratedParams}
            />

            <ParamInputs
              modelType={modelType}
              params={model.params}
              onChange={model.setParams}
              onReset={model.reset}
              canReset={model.isCalibrated}
            />

            {!model.isCalibrated && (
              <p className="text-[11px] text-warn">
                No calibrated {modelType} params — these are hand-set defaults.
                Calibrate a ticker to replace them.
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 self-start lg:grid-cols-3">
            <Field label="S₀" hint="spot">
              <NumberInput value={S0} min={0.01} step={1} onChange={setS0} />
            </Field>
            <Field label="T" hint="years">
              <NumberInput value={T} min={0.01} step={0.25} onChange={setT} />
            </Field>
            <Field label="dt" hint="years/step">
              <NumberInput value={dt} min={0.0001} step={0.001} onChange={setDt} />
            </Field>
            <Field label="Paths">
              <NumberInput
                value={nSimulations}
                min={1}
                step={1000}
                onChange={setNSimulations}
              />
            </Field>
            <Field label="Variance reduction">
              <Select
                value={varianceReduction}
                onChange={setVarianceReduction}
                options={VARIANCE_REDUCTION}
              />
            </Field>
            <Field label="Seed" hint="optional">
              <NumberInput value={seed} step={1} placeholder="random" onChange={setSeed} />
            </Field>

            {tooManyPaths && (
              <p className="col-span-full text-[11px] text-bad">
                The API caps a request at {MAX_SIMULATIONS.toLocaleString()} paths
                and will answer 413 above that.
              </p>
            )}
            <p className="col-span-full text-[11px] text-ink-faint">
              dt = {(1 / dt).toFixed(0)} steps per year · {Math.round(T / dt)} steps
              over the horizon. A seed makes the run — and the sample paths
              picked out of it — reproducible.
            </p>
          </div>
        </div>

        <div className="mt-5 border-t border-edge pt-5">
          {simulation.error && <ErrorBanner message={simulation.error} />}

          {simulation.loading && <LoadingPanel label="Drawing paths…" />}

          {!simulation.loading && !simulation.chart && !simulation.error && (
            <EmptyState>
              Run a simulation to see the confidence cone — percentile bands
              across every path, with a handful of individual paths drawn
              through them.
            </EmptyState>
          )}

          {!simulation.loading && simulation.chart && (
            <div className="space-y-4">
              <ConfidenceCone chart={simulation.chart} />

              <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
                <Stat label="Mean" value={fmtMoney(summary.mean)} tone="accent" />
                <Stat label="Median" value={fmtMoney(summary.median)} />
                <Stat label="Std dev" value={fmtMoney(summary.std)} />
                <Stat label="Min" value={fmtMoney(summary.min)} />
                <Stat label="Max" value={fmtMoney(summary.max)} />
              </div>

              <p className="text-[11px] text-ink-faint">
                Terminal values over {simulation.data.n_simulations.toLocaleString()} paths
                · computed in {fmtMillis(simulation.data.computation_time_ms)} · bands
                downsampled to {simulation.data.time_axis.length} dates for transport.
              </p>
            </div>
          )}
        </div>
      </Card>

      {/* ---------------------------------------------------------------- */}
      <Card
        title="Price an option"
        subtitle="Risk-neutral measure — the drift is overridden with r − q."
        right={
          <Button onClick={onPrice} disabled={pricing.loading || tooManyPaths}>
            {pricing.loading ? (
              <LoadingSpinner label="Pricing…" />
            ) : (
              <>
                <Sigma size={15} /> Price
              </>
            )}
          </Button>
        }
      >
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          <Field label="Option type">
            <Select
              value={optionType}
              onChange={setOptionType}
              options={OPTION_TYPES.map((o) => ({ key: o.key, label: o.label }))}
            />
          </Field>
          <Field label="Strike K">
            <NumberInput
              value={spec.needsStrike ? K : ""}
              min={0.01}
              step={1}
              disabled={!spec.needsStrike}
              placeholder="floating"
              onChange={setK}
            />
          </Field>
          <Field label="r" hint="risk-free">
            <NumberInput value={r} step={0.005} onChange={setR} />
          </Field>
          <Field label="q" hint="dividend yield">
            <NumberInput value={q} step={0.005} onChange={setQ} />
          </Field>
          <Field label="Barrier">
            <NumberInput
              value={spec.needsBarrier ? barrier : ""}
              min={0.01}
              step={1}
              disabled={!spec.needsBarrier}
              placeholder="n/a"
              onChange={setBarrier}
            />
          </Field>
        </div>

        <p className="mt-3 text-[11px] text-ink-faint">
          Uses the model, params, horizon and path count set above. Greeks are
          bump-and-revalue with common random numbers, so pricing costs roughly
          ten simulations rather than one.
        </p>

        <div className="mt-5 border-t border-edge pt-5">
          {pricing.error && <ErrorBanner message={pricing.error} />}

          {pricing.loading && <LoadingPanel label="Pricing and bumping…" />}

          {!pricing.loading && !priced && !pricing.error && (
            <EmptyState>
              Price the option to see its value, Greeks and payoff distribution.
            </EmptyState>
          )}

          {!pricing.loading && priced && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <Stat
                  label="Price"
                  value={fmtMoney(priced.price)}
                  tone="accent"
                  hint={`± ${fmtMoney(priced.std_error, 4)} std err`}
                />
                <Stat
                  label="95% CI"
                  value={`${fmtMoney(priced.confidence_interval_95[0])} – ${fmtMoney(
                    priced.confidence_interval_95[1],
                  )}`}
                />
                {priced.bs_benchmark ? (
                  <>
                    <Stat
                      label="Black-Scholes"
                      value={fmtMoney(priced.bs_benchmark.price)}
                      hint="closed form"
                    />
                    <Stat
                      label="Relative error"
                      value={
                        priced.bs_benchmark.relative_error === null
                          ? "—"
                          : fmtPercent(priced.bs_benchmark.relative_error)
                      }
                      tone={
                        (priced.bs_benchmark.relative_error ?? 0) < 0.02 ? "good" : "warn"
                      }
                      hint="vs closed form"
                    />
                  </>
                ) : (
                  <Stat
                    label="Black-Scholes"
                    value="n/a"
                    hint="vanillas under GBM only"
                  />
                )}
              </div>

              <div className="grid gap-5 xl:grid-cols-2">
                <div>
                  <h3 className="mb-3 text-xs font-medium tracking-wide text-ink-dim uppercase">
                    Greeks
                  </h3>
                  <GreeksTable
                    greeks={priced.greeks}
                    benchmark={priced.bs_benchmark?.greeks}
                  />
                </div>

                <div>
                  <h3 className="mb-3 text-xs font-medium tracking-wide text-ink-dim uppercase">
                    Payoff distribution
                  </h3>
                  <PayoffHistogram data={pricing.histogram} />
                  <p className="mt-1 text-[11px] text-ink-faint">
                    Undiscounted payoff per path, 50 bins. The spike at zero is
                    every path that expired worthless.
                  </p>
                </div>
              </div>

              <p className="text-[11px] text-ink-faint">
                {priced.n_simulations.toLocaleString()} paths ·{" "}
                {fmtMillis(priced.computation_time_ms)}
                {priced.mlflow_run_id && ` · MLflow run ${priced.mlflow_run_id}`}
                {priced.bs_benchmark === null &&
                  " · no closed-form benchmark for this payoff/model pair"}
              </p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}

export default SimulatePricePage;
