import { ArrowRight } from "lucide-react";
import { fmtMoney, fmtPercent } from "../lib/format";

/**
 * Baseline against stressed, metric by metric.
 *
 * Both runs share a seed, so the delta column is the shock alone rather than
 * the difference between two independent Monte Carlo samples.
 *
 * "Worse" is not the same direction for every row — a bigger drawdown is bad,
 * a bigger Sharpe is good, and VaR is negative to begin with — so each row
 * declares which way is bad and the colouring follows from that instead of
 * from the sign of the change.
 */

function rowsFor(metrics, asPercent) {
  const money = (v) => (asPercent ? fmtPercent(v) : fmtMoney(v));
  const levels = Object.keys(metrics.var).sort(
    (a, b) => Number.parseFloat(a) - Number.parseFloat(b),
  );

  return [
    ...levels.map((level) => ({
      key: `var-${level}`,
      label: `VaR ${level}%`,
      pick: (m) => m.var[level],
      format: money,
      worseWhen: "lower",
    })),
    ...levels.map((level) => ({
      key: `cvar-${level}`,
      label: `CVaR ${level}%`,
      pick: (m) => m.cvar[level],
      format: money,
      worseWhen: "lower",
    })),
    {
      key: "dd-mean",
      label: "Max drawdown (mean)",
      pick: (m) => m.max_drawdown.mean,
      format: fmtPercent,
      worseWhen: "higher",
    },
    {
      key: "dd-95",
      label: "Max drawdown (95th)",
      pick: (m) => m.max_drawdown.percentile_95,
      format: fmtPercent,
      worseWhen: "higher",
    },
    {
      key: "sharpe",
      label: "Sharpe",
      pick: (m) => m.sharpe_ratio,
      format: (v) => v.toFixed(3),
      worseWhen: "lower",
    },
    {
      key: "ploss",
      label: "P(loss)",
      pick: (m) => m.probability_of_loss,
      format: fmtPercent,
      worseWhen: "higher",
    },
  ];
}

export function ScenarioComparison({ baseline, stressed, scenarioName, basis = "currency" }) {
  if (!baseline || !stressed) return null;

  const rows = rowsFor(baseline, basis === "fraction");

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] text-sm">
        <thead>
          <tr className="border-b border-edge text-[11px] tracking-wide text-ink-dim uppercase">
            <th className="py-2 text-left font-medium">Metric</th>
            <th className="py-2 text-right font-medium">Baseline</th>
            <th className="py-2 text-right font-medium">
              <span className="inline-flex items-center gap-1.5 text-warn">
                <ArrowRight size={12} />
                {scenarioName ?? "Stressed"}
              </span>
            </th>
            <th className="py-2 text-right font-medium">Change</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-edge/70">
          {rows.map((row) => {
            const before = row.pick(baseline);
            const after = row.pick(stressed);
            const delta = after - before;
            const worse = row.worseWhen === "lower" ? delta < 0 : delta > 0;
            const material = Math.abs(delta) > 1e-9;

            return (
              <tr key={row.key}>
                <td className="py-2 text-ink">{row.label}</td>
                <td className="tnum py-2 text-right text-ink-dim">{row.format(before)}</td>
                <td className="tnum py-2 text-right font-medium text-ink">
                  {row.format(after)}
                </td>
                <td
                  className={`tnum py-2 text-right text-xs ${
                    !material ? "text-ink-faint" : worse ? "text-bad" : "text-good"
                  }`}
                >
                  {delta > 0 ? "+" : ""}
                  {row.format(delta)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default ScenarioComparison;
