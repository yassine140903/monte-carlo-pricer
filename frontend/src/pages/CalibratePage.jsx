import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, Play, XCircle } from "lucide-react";
import { getTickerData } from "../api/client";
import ParamDisplay from "../components/ParamDisplay";
import TickerSelector from "../components/TickerSelector";
import { LoadingPanel, LoadingSpinner } from "../components/LoadingSpinner";
import { Badge, Button, Card, EmptyState, ErrorBanner, Field, NumberInput } from "../components/ui";
import { useApp } from "../context/AppContext";
import useCalibration from "../hooks/useCalibration";
import { fmt } from "../lib/format";
import { MODELS, MODEL_KEYS } from "../lib/models";

const FIT_ROWS = [
  { key: "log_likelihood", label: "Log-likelihood" },
  { key: "aic", label: "AIC" },
  { key: "bic", label: "BIC" },
];

function ModelCard({ modelType, result }) {
  const model = MODELS[modelType];

  return (
    <Card
      title={model.label}
      subtitle={model.fullName}
      right={
        result.status === "success" ? (
          <Badge tone="good">
            <CheckCircle2 size={11} /> fitted
          </Badge>
        ) : result.status === "error" ? (
          <Badge tone="bad">
            <XCircle size={11} /> failed
          </Badge>
        ) : result.status === "loading" ? (
          <LoadingSpinner />
        ) : (
          <Badge>idle</Badge>
        )
      }
    >
      {result.status === "loading" && <LoadingPanel label="Fitting…" minHeight="min-h-48" />}

      {result.status === "error" && <ErrorBanner message={result.error} />}

      {result.status === "idle" && (
        <p className="py-8 text-center text-xs text-ink-faint">Not calibrated yet.</p>
      )}

      {result.status === "success" && (
        <div className="space-y-4">
          <ParamDisplay modelType={modelType} params={result.data.params} />

          <div className="rounded-lg border border-edge bg-surface-2 px-3.5 py-3">
            <p className="mb-2 text-[11px] font-medium tracking-wide text-ink-dim uppercase">
              Goodness of fit
            </p>
            <dl className="space-y-1">
              {FIT_ROWS.map(({ key, label }) => (
                <div key={key} className="flex items-baseline justify-between gap-3">
                  <dt className="text-xs text-ink-faint">{label}</dt>
                  <dd className="tnum text-xs font-medium text-ink">
                    {fmt(result.data.fit_metrics[key], 2)}
                  </dd>
                </div>
              ))}
            </dl>
            <p className="mt-2 border-t border-edge pt-2 text-[11px] text-ink-faint">
              Lower AIC/BIC is a better fit, but only comparably so — all three
              are fitted to the same log returns except Heston, which fits a
              realized-volatility term structure instead.
            </p>
          </div>

          {result.data.mlflow_run_id && (
            <p className="tnum truncate text-[11px] text-ink-faint">
              MLflow run {result.data.mlflow_run_id}
            </p>
          )}
        </div>
      )}
    </Card>
  );
}

export function CalibratePage() {
  const { selectedTicker, setSelectedTicker, setS0 } = useApp();
  const { results, loading, run } = useCalibration();

  const [lookbackDays, setLookbackDays] = useState(756);
  const [windowDays, setWindowDays] = useState("");
  const [meta, setMeta] = useState(null);
  const [spot, setSpot] = useState(null);

  const anyResult = MODEL_KEYS.some((key) => results[key].status !== "idle");

  /**
   * Adopt the ticker's most recent close as S0 for the whole app.
   *
   * Requesting the single day the metadata reports as `latest_date` rather
   * than the whole series — the other pages need one number, and the full
   * history is a few hundred kilobytes of JSON to get it.
   */
  const onMeta = useCallback((entry) => setMeta(entry), []);

  useEffect(() => {
    if (!meta) return;
    let active = true;

    getTickerData(meta.ticker, meta.latest_date, meta.latest_date)
      .then((response) => {
        const bars = response.data.bars;
        if (!active || bars.length === 0) return;
        const close = Number(bars[bars.length - 1].close.toFixed(2));
        setSpot(close);
        setS0(close);
      })
      // A missing spot is not worth an error banner: every page that needs
      // S0 exposes it as an editable field with a working default.
      .catch(() => active && setSpot(null));

    return () => {
      active = false;
    };
  }, [meta, setS0]);

  const onCalibrate = () => {
    if (!selectedTicker) return;
    run(selectedTicker, {
      lookbackDays: Number(lookbackDays) || 756,
      windowDays: windowDays === "" ? null : Number(windowDays),
    });
  };

  return (
    <div className="space-y-5">
      <Card
        title="Calibrate"
        subtitle="Fit all three model families to one ticker's stored history."
      >
        <div className="grid items-end gap-4 md:grid-cols-[minmax(0,1fr)_160px_160px_auto]">
          <TickerSelector
            value={selectedTicker}
            onChange={setSelectedTicker}
            onMeta={onMeta}
            disabled={loading}
          />

          <Field label="Lookback" hint="trading days">
            <NumberInput
              value={lookbackDays}
              min={30}
              step={63}
              disabled={loading}
              onChange={setLookbackDays}
            />
          </Field>

          <Field label="GBM window" hint="optional">
            <NumberInput
              value={windowDays}
              min={30}
              step={21}
              placeholder="full lookback"
              disabled={loading}
              onChange={setWindowDays}
            />
          </Field>

          <Button onClick={onCalibrate} disabled={loading || !selectedTicker}>
            {loading ? <LoadingSpinner label="Calibrating…" /> : <><Play size={15} /> Calibrate</>}
          </Button>
        </div>

        <p className="mt-3 text-[11px] text-ink-faint">
          {spot !== null && (
            <>
              Latest close <span className="tnum text-ink-dim">{spot}</span> — carried
              to the other pages as S₀.{" "}
            </>
          )}
          756 trading days ≈ 3 years. The GBM window narrows that fit further to
          the most recent stretch — a shorter window tracks the current
          volatility regime more closely at the cost of a noisier estimate. The
          other two models fit a likelihood or a term structure over the whole
          sample and have no equivalent.
        </p>
      </Card>

      {!anyResult ? (
        <EmptyState>
          Pick a ticker and calibrate. All three models are fitted in parallel,
          and whatever succeeds flows through to the Simulate and Risk pages.
        </EmptyState>
      ) : (
        <div className="grid gap-4 lg:grid-cols-3">
          {MODEL_KEYS.map((modelType) => (
            <ModelCard key={modelType} modelType={modelType} result={results[modelType]} />
          ))}
        </div>
      )}
    </div>
  );
}

export default CalibratePage;
