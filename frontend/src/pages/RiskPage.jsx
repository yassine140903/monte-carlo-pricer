import { useMemo, useState } from "react";
import { Play, Plus, Trash2 } from "lucide-react";
import ModelSelector from "../components/ModelSelector";
import ParamInputs from "../components/ParamInputs";
import RiskGauges from "../components/RiskGauges";
import ScenarioComparison from "../components/ScenarioComparison";
import { LoadingPanel, LoadingSpinner } from "../components/LoadingSpinner";
import {
  Button,
  Card,
  EmptyState,
  ErrorBanner,
  Field,
  NumberInput,
  Select,
  Toggle,
} from "../components/ui";
import { useApp } from "../context/AppContext";
import useModelParams from "../hooks/useModelParams";
import { usePortfolioRisk, useScenarios, useSingleAssetRisk } from "../hooks/useRisk";
import { fmt } from "../lib/format";
import { MODELS, MODEL_KEYS, MAX_SIMULATIONS, defaultParams } from "../lib/models";

const MODES = [
  { key: "single", label: "Single asset" },
  { key: "portfolio", label: "Portfolio" },
];

const CONFIDENCE_PRESETS = [
  { key: "95,99", label: "95% and 99%" },
  { key: "95", label: "95% only" },
  { key: "90,95,99", label: "90%, 95% and 99%" },
];

const parseLevels = (key) => key.split(",").map((p) => Number(p) / 100);

// ---------------------------------------------------------------------------

function SingleAssetMode() {
  const { calibratedParams, S0, setS0 } = useApp();
  const [modelType, setModelType] = useState("gbm");
  const model = useModelParams(modelType);

  const [T, setT] = useState(1);
  const [nSimulations, setNSimulations] = useState(10000);
  const [levelsKey, setLevelsKey] = useState("95,99");
  const [r, setR] = useState(0.05);
  const [seed, setSeed] = useState(42);
  const [scenarioKey, setScenarioKey] = useState("");

  const { forModel, loading: scenariosLoading } = useScenarios();
  const risk = useSingleAssetRisk();

  // A scenario names parameters, so only the families with a variant for the
  // selected model are offered — `v0` means nothing to GBM.
  const available = forModel(modelType);
  const scenarioOptions = useMemo(
    () => [
      { key: "", label: "No scenario (baseline only)" },
      ...available.map((family) => ({ key: family.key, label: family.name })),
    ],
    [available],
  );

  // Switching model can invalidate the chosen scenario.
  const effectiveScenario = available.some((f) => f.key === scenarioKey) ? scenarioKey : "";
  const scenarioName = available.find((f) => f.key === effectiveScenario)?.name;

  const tooManyPaths = nSimulations > MAX_SIMULATIONS;

  const onCompute = () =>
    risk.run(
      {
        S0: Number(S0),
        model_params: model.body,
        T: Number(T),
        n_simulations: Number(nSimulations),
        confidence_levels: parseLevels(levelsKey),
        r: Number(r),
        seed: seed === "" ? null : Number(seed),
      },
      effectiveScenario || null,
    );

  return (
    <Card
      title="Single asset risk"
      subtitle="Physical measure — r is the Sharpe benchmark only and never reaches the simulator."
      right={
        <Button onClick={onCompute} disabled={risk.loading || tooManyPaths}>
          {risk.loading ? (
            <LoadingSpinner label="Computing…" />
          ) : (
            <>
              <Play size={15} /> Compute
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
        </div>

        <div className="grid grid-cols-2 gap-3 self-start lg:grid-cols-3">
          <Field label="S₀" hint="spot">
            <NumberInput value={S0} min={0.01} step={1} onChange={setS0} />
          </Field>
          <Field label="T" hint="years">
            <NumberInput value={T} min={0.01} step={0.25} onChange={setT} />
          </Field>
          <Field label="Paths">
            <NumberInput value={nSimulations} min={1} step={1000} onChange={setNSimulations} />
          </Field>
          <Field label="Confidence">
            <Select value={levelsKey} onChange={setLevelsKey} options={CONFIDENCE_PRESETS} />
          </Field>
          <Field label="r" hint="Sharpe benchmark">
            <NumberInput value={r} step={0.005} onChange={setR} />
          </Field>
          <Field label="Seed" hint="optional">
            <NumberInput value={seed} step={1} placeholder="random" onChange={setSeed} />
          </Field>

          <div className="col-span-full">
            <Field label="Stress scenario" hint={scenariosLoading ? "loading…" : ""}>
              <Select
                value={effectiveScenario}
                onChange={setScenarioKey}
                options={scenarioOptions}
                disabled={scenariosLoading || available.length === 0}
              />
            </Field>
            <p className="mt-1.5 text-[11px] text-ink-faint">
              A scenario shocks the calibrated params and re-runs on the same
              seed, so the difference is the shock rather than Monte Carlo noise.
            </p>
          </div>

          {tooManyPaths && (
            <p className="col-span-full text-[11px] text-bad">
              The API caps a request at {MAX_SIMULATIONS.toLocaleString()} paths.
            </p>
          )}
        </div>
      </div>

      <div className="mt-5 border-t border-edge pt-5">
        {risk.error && <ErrorBanner message={risk.error} />}
        {risk.loading && <LoadingPanel label="Simulating and measuring…" />}

        {!risk.loading && !risk.baseline && !risk.error && (
          <EmptyState>
            Compute to see VaR, CVaR, drawdown, Sharpe and probability of loss.
          </EmptyState>
        )}

        {!risk.loading && risk.baseline && (
          <div className="space-y-5">
            <RiskGauges metrics={risk.baseline} />

            {risk.stressed && (
              <div>
                <h3 className="mb-3 text-xs font-medium tracking-wide text-ink-dim uppercase">
                  Baseline vs {scenarioName}
                </h3>
                <ScenarioComparison
                  baseline={risk.baseline}
                  stressed={risk.stressed}
                  scenarioName={scenarioName}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------

const blankAsset = (ticker = "") => ({
  id: crypto.randomUUID(),
  ticker,
  weight: 0.5,
  S0: 100,
  modelType: "gbm",
});

function PortfolioMode() {
  const { calibratedParams, selectedTicker, S0 } = useApp();

  const [assets, setAssets] = useState(() => [
    { ...blankAsset(selectedTicker ?? ""), weight: 0.5, S0: S0 },
    { ...blankAsset(""), weight: 0.5, S0: 100 },
  ]);
  const [T, setT] = useState(1);
  const [nSimulations, setNSimulations] = useState(10000);
  const [levelsKey, setLevelsKey] = useState("95,99");
  const [lookbackDays, setLookbackDays] = useState(756);
  const [r, setR] = useState(0.05);
  const [seed, setSeed] = useState(42);
  const [scenarioKey, setScenarioKey] = useState("");

  const { scenarios, loading: scenariosLoading } = useScenarios();
  const portfolio = usePortfolioRisk();

  const totalWeight = assets.reduce((sum, a) => sum + (Number(a.weight) || 0), 0);
  const weightsOk = Math.abs(totalWeight - 1) < 1e-6;
  const tickersOk = assets.every((a) => a.ticker.trim().length > 0);
  const tooManyPaths = nSimulations > MAX_SIMULATIONS;

  // Only offer scenarios that cover every model family in the basket — a
  // preset with no variant for one asset would 404 the whole request.
  const usedModels = [...new Set(assets.map((a) => a.modelType))];
  const scenarioOptions = [
    { key: "", label: "No scenario" },
    ...scenarios
      .filter((family) => usedModels.every((m) => family.model_types.includes(m)))
      .map((family) => ({ key: family.key, label: family.name })),
  ];

  const update = (id, patch) =>
    setAssets((current) => current.map((a) => (a.id === id ? { ...a, ...patch } : a)));

  const onCompute = () =>
    portfolio.run({
      assets: assets.map((asset) => ({
        ticker: asset.ticker.trim().toUpperCase(),
        weight: Number(asset.weight),
        S0: Number(asset.S0),
        // Calibrated params where the ticker on this row happens to be the one
        // that was calibrated; otherwise the hand-set defaults for its family.
        model_params:
          asset.ticker.trim().toUpperCase() === selectedTicker &&
          calibratedParams[asset.modelType]
            ? calibratedParams[asset.modelType]
            : defaultParams(asset.modelType),
      })),
      T: Number(T),
      n_simulations: Number(nSimulations),
      confidence_levels: parseLevels(levelsKey),
      lookback_days: Number(lookbackDays),
      scenario: scenarioKey || null,
      r: Number(r),
      seed: seed === "" ? null : Number(seed),
    });

  const result = portfolio.data;

  return (
    <Card
      title="Portfolio risk"
      subtitle="Correlations estimated from stored history; the forward simulation uses the params below."
      right={
        <Button
          onClick={onCompute}
          disabled={portfolio.loading || !weightsOk || !tickersOk || tooManyPaths}
        >
          {portfolio.loading ? (
            <LoadingSpinner label="Computing…" />
          ) : (
            <>
              <Play size={15} /> Compute
            </>
          )}
        </Button>
      }
    >
      <div className="space-y-3">
        {assets.map((asset, index) => (
          <div
            key={asset.id}
            className="grid items-end gap-3 rounded-lg border border-edge bg-surface-2 p-3
              md:grid-cols-[minmax(0,200px)_110px_120px_minmax(0,180px)_auto]
              md:justify-start"
          >
            <Field label={`Asset ${index + 1} · ticker`}>
              <input
                className="w-full rounded-lg border border-edge bg-surface px-3 py-2 text-sm
                  text-ink uppercase outline-none transition focus:border-accent
                  focus:ring-1 focus:ring-accent"
                value={asset.ticker}
                placeholder="AAPL"
                onChange={(e) => update(asset.id, { ticker: e.target.value })}
              />
            </Field>
            <Field label="Weight">
              <NumberInput
                value={asset.weight}
                step={0.05}
                onChange={(v) => update(asset.id, { weight: v })}
              />
            </Field>
            <Field label="S₀">
              <NumberInput
                value={asset.S0}
                min={0.01}
                step={1}
                onChange={(v) => update(asset.id, { S0: v })}
              />
            </Field>
            <Field label="Model">
              <Select
                value={asset.modelType}
                onChange={(v) => update(asset.id, { modelType: v })}
                options={MODEL_KEYS.map((k) => ({ key: k, label: MODELS[k].label }))}
              />
            </Field>
            <button
              type="button"
              onClick={() => setAssets((c) => c.filter((a) => a.id !== asset.id))}
              disabled={assets.length <= 2}
              title={assets.length <= 2 ? "A portfolio needs at least two assets" : "Remove"}
              className="mb-0.5 rounded-lg border border-edge px-3 py-2 text-ink-faint
                transition hover:border-bad hover:text-bad disabled:cursor-not-allowed
                disabled:opacity-40"
            >
              <Trash2 size={15} />
            </button>
          </div>
        ))}

        <div className="flex flex-wrap items-center gap-3">
          <Button variant="ghost" onClick={() => setAssets((c) => [...c, blankAsset()])}>
            <Plus size={15} /> Add asset
          </Button>
          <span
            className={`tnum text-xs ${weightsOk ? "text-ink-faint" : "text-warn"}`}
          >
            weights sum to {fmt(totalWeight, 3)}
            {!weightsOk && " — must be 1.0"}
          </span>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-4 xl:grid-cols-7">
        <Field label="T" hint="years">
          <NumberInput value={T} min={0.01} step={0.25} onChange={setT} />
        </Field>
        <Field label="Paths">
          <NumberInput value={nSimulations} min={1} step={1000} onChange={setNSimulations} />
        </Field>
        <Field label="Confidence">
          <Select value={levelsKey} onChange={setLevelsKey} options={CONFIDENCE_PRESETS} />
        </Field>
        <Field label="Lookback" hint="corr. window">
          <NumberInput value={lookbackDays} min={30} step={63} onChange={setLookbackDays} />
        </Field>
        <Field label="r" hint="Sharpe benchmark">
          <NumberInput value={r} step={0.005} onChange={setR} />
        </Field>
        <Field label="Seed" hint="optional">
          <NumberInput value={seed} step={1} placeholder="random" onChange={setSeed} />
        </Field>
        <Field label="Scenario" hint={scenariosLoading ? "loading…" : ""}>
          <Select
            value={scenarioKey}
            onChange={setScenarioKey}
            options={scenarioOptions}
            disabled={scenariosLoading}
          />
        </Field>
      </div>

      <p className="mt-3 text-[11px] text-ink-faint">
        Every ticker must already be in the database — correlations come from
        the overlap of their stored trading days. Paths start from a normalized
        value of 1.0, so portfolio P&amp;L reads as a fraction of capital.
      </p>

      <div className="mt-5 border-t border-edge pt-5">
        {portfolio.error && <ErrorBanner message={portfolio.error} />}
        {portfolio.loading && <LoadingPanel label="Correlating and simulating…" />}

        {!portfolio.loading && !result && !portfolio.error && (
          <EmptyState>
            Add at least two stored tickers with weights summing to 1.0, then
            compute.
          </EmptyState>
        )}

        {!portfolio.loading && result && (
          <div className="space-y-5">
            <RiskGauges metrics={result.risk_metrics} basis="fraction" />

            <div>
              <h3 className="mb-3 text-xs font-medium tracking-wide text-ink-dim uppercase">
                Estimated correlation
                {result.scenario_applied && ` · stressed: ${result.scenario_applied}`}
              </h3>
              <div className="overflow-x-auto">
                <table className="text-sm">
                  <thead>
                    <tr>
                      <th className="px-3 py-1.5" />
                      {result.tickers.map((ticker) => (
                        <th
                          key={ticker}
                          className="px-3 py-1.5 text-xs font-medium text-ink-dim"
                        >
                          {ticker}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.correlation_matrix.map((row, i) => (
                      <tr key={result.tickers[i]}>
                        <th className="px-3 py-1.5 text-left text-xs font-medium text-ink-dim">
                          {result.tickers[i]}
                        </th>
                        {row.map((value, j) => (
                          <td
                            key={`${i}-${j}`}
                            className="tnum px-3 py-1.5 text-center text-xs"
                            style={{
                              // Correlation shaded on the accent hue: the
                              // structure is readable at a glance without a
                              // second colour scale to interpret.
                              background: `color-mix(in srgb, var(--color-accent) ${
                                Math.max(0, value) * 28
                              }%, transparent)`,
                            }}
                          >
                            {value.toFixed(3)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-[11px] text-ink-faint">
                Rows and columns are in the alphabetical order the estimator
                returns; the simulation was run in that same order.
              </p>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------

export function RiskPage() {
  const [mode, setMode] = useState("single");

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between gap-4">
        <Toggle value={mode} onChange={setMode} options={MODES} />
      </div>

      {mode === "single" ? <SingleAssetMode /> : <PortfolioMode />}
    </div>
  );
}

export default RiskPage;
