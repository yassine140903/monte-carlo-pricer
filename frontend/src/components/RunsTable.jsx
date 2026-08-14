import { ExternalLink } from "lucide-react";
import { MLFLOW_UI_URL } from "../api/client";
import { fmtCount } from "../lib/format";
import { EmptyState } from "./ui";

/**
 * The MLflow experiment listing.
 *
 * Intentionally shallow, matching the endpoint: /mlflow/experiments returns
 * name, id and run count and nothing else. Browsing individual runs, comparing
 * them and reading artifacts is what the MLflow UI already does well, so each
 * row links out to it rather than reimplementing it here.
 */
export function RunsTable({ experiments }) {
  if (!experiments?.length) {
    return (
      <EmptyState>
        No experiments logged yet. Calibrate a ticker or price an option and
        they will appear here — unless the tracking server is unreachable, in
        which case the API returns an empty list rather than an error.
      </EmptyState>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] text-sm">
        <thead>
          <tr className="border-b border-edge text-[11px] tracking-wide text-ink-dim uppercase">
            <th className="py-2 text-left font-medium">Experiment</th>
            <th className="py-2 text-left font-medium">ID</th>
            <th className="py-2 text-right font-medium">Runs</th>
            <th className="py-2 text-right font-medium">Open</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-edge/70">
          {experiments.map((experiment) => (
            <tr key={experiment.experiment_id} className="hover:bg-surface-2/60">
              <td className="py-2.5 font-medium text-ink">{experiment.name}</td>
              <td className="tnum py-2.5 text-xs text-ink-faint">
                {experiment.experiment_id}
              </td>
              <td className="tnum py-2.5 text-right text-ink-dim">
                {fmtCount(experiment.run_count)}
              </td>
              <td className="py-2.5 text-right">
                <a
                  href={`${MLFLOW_UI_URL}/#/experiments/${experiment.experiment_id}`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
                >
                  MLflow
                  <ExternalLink size={12} />
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default RunsTable;
