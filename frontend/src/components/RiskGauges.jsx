import { fmtMoney, fmtPercent } from "../lib/format";
import { Stat } from "./ui";

/**
 * The risk metric grid.
 *
 * VaR and CVaR arrive as raw P&L quantiles — a loss is *negative*, the
 * convention src/risk/metrics.py documents. That sign is kept rather than
 * flipped: showing "VaR 22.71" next to "CVaR 28.97" invites reading the bigger
 * number as the better one, when both are losses and CVaR is always the worse.
 *
 * `basis` scales the display. A single asset is priced in currency; a
 * portfolio is simulated from a normalized start of 1.0, so its P&L is a
 * fraction of capital and reads better as a percentage.
 */
export function RiskGauges({ metrics, basis = "currency" }) {
  if (!metrics) return null;

  const asPercent = basis === "fraction";
  const money = (v) => (asPercent ? fmtPercent(v) : fmtMoney(v));

  const levels = Object.keys(metrics.var).sort(
    (a, b) => Number.parseFloat(a) - Number.parseFloat(b),
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {levels.map((level) => (
          <Stat
            key={`var-${level}`}
            label={`VaR ${level}%`}
            value={money(metrics.var[level])}
            tone="bad"
            hint={`worst ${(100 - Number.parseFloat(level)).toFixed(0)}% cutoff`}
          />
        ))}
        {levels.map((level) => (
          <Stat
            key={`cvar-${level}`}
            label={`CVaR ${level}%`}
            value={money(metrics.cvar[level])}
            tone="bad"
            hint="mean loss beyond VaR"
          />
        ))}
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat
          label="Max drawdown"
          value={fmtPercent(metrics.max_drawdown.mean)}
          hint="mean across paths"
        />
        <Stat
          label="Drawdown 95th"
          value={fmtPercent(metrics.max_drawdown.percentile_95)}
          tone="warn"
          hint="tail of the drawdown"
        />
        <Stat
          label="Sharpe"
          value={metrics.sharpe_ratio.toFixed(3)}
          tone={metrics.sharpe_ratio >= 0 ? "good" : "bad"}
          hint="excess return / vol"
        />
        <Stat
          label="P(loss)"
          value={fmtPercent(metrics.probability_of_loss)}
          hint="finishes below start"
        />
      </div>

      <p className="text-[11px] text-ink-faint">
        Signed P&amp;L quantiles: losses are negative, and CVaR is always at
        least as bad as VaR. Simulated under the physical measure — the
        calibrated drift, not <span className="tnum">r − q</span>.
      </p>
    </div>
  );
}

export default RiskGauges;
