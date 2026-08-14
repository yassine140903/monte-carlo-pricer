import { useCallback, useEffect, useState } from "react";
import { ExternalLink, RefreshCw } from "lucide-react";
import RunsTable from "../components/RunsTable";
import { LoadingPanel } from "../components/LoadingSpinner";
import { Button, Card, ErrorBanner, Stat } from "../components/ui";
import { MLFLOW_UI_URL, apiErrorMessage, getMlflowExperiments } from "../api/client";

export function RunsPage() {
  const [experiments, setExperiments] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);

    getMlflowExperiments()
      .then((response) => setExperiments(response.data.experiments))
      .catch((err) => setError(apiErrorMessage(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  const totalRuns = experiments?.reduce((sum, e) => sum + e.run_count, 0) ?? 0;

  return (
    <div className="space-y-5">
      <Card
        title="MLflow experiments"
        subtitle="Calibrations and pricings logged by the API."
        right={
          <div className="flex items-center gap-2">
            <Button variant="ghost" onClick={load} disabled={loading}>
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
              Refresh
            </Button>
            <a
              href={MLFLOW_UI_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-lg border border-edge
                bg-surface-2 px-4 py-2 text-sm text-ink transition hover:border-accent
                hover:text-accent"
            >
              Open MLflow
              <ExternalLink size={14} />
            </a>
          </div>
        }
      >
        {error && <ErrorBanner message={error} />}

        {loading && <LoadingPanel label="Reading the tracking server…" minHeight="min-h-40" />}

        {!loading && experiments && (
          <div className="space-y-5">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <Stat label="Experiments" value={experiments.length} />
              <Stat label="Total runs" value={totalRuns} tone="accent" />
              <Stat label="Tracking server" value="localhost:5000" />
            </div>

            <RunsTable experiments={experiments} />

            <p className="text-[11px] leading-relaxed text-ink-faint">
              This page is intentionally shallow — the endpoint returns names and
              run counts, nothing more. Comparing runs, plotting metrics and
              reading artifacts is what the MLflow UI does well, so each row
              links straight into it. If the tracking server is down the API
              returns an empty list rather than an error, so an empty table here
              means either no runs yet or no server.
            </p>
          </div>
        )}
      </Card>
    </div>
  );
}

export default RunsPage;
