import { fmt } from "../lib/format";

/**
 * Delta / Gamma / Vega / Theta / Rho, with the closed-form column beside them
 * where the backend supplied one.
 *
 * Units follow src/pricing/black_scholes.py: vega is per 1.0 of volatility,
 * theta per year, rho per 1.0 of rate. The hints spell that out because a
 * theta of -6 reads alarmingly until you know it is annual.
 */

const GREEKS = [
  { key: "delta", label: "Delta", symbol: "Δ", hint: "∂V/∂S" },
  { key: "gamma", label: "Gamma", symbol: "Γ", hint: "∂²V/∂S²" },
  { key: "vega", label: "Vega", symbol: "ν", hint: "per 1.0 vol" },
  { key: "theta", label: "Theta", symbol: "Θ", hint: "per year" },
  { key: "rho", label: "Rho", symbol: "ρ", hint: "per 1.0 rate" },
];

export function GreeksTable({ greeks, benchmark }) {
  if (!greeks) return null;
  const hasBenchmark = Boolean(benchmark);

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b border-edge text-[11px] tracking-wide text-ink-dim uppercase">
          <th className="py-2 text-left font-medium">Greek</th>
          <th className="py-2 text-right font-medium">Monte Carlo</th>
          {hasBenchmark && <th className="py-2 text-right font-medium">Black-Scholes</th>}
          {hasBenchmark && <th className="py-2 text-right font-medium">Diff</th>}
        </tr>
      </thead>
      <tbody className="divide-y divide-edge/70">
        {GREEKS.map(({ key, label, symbol, hint }) => {
          const mc = greeks[key];
          const bs = benchmark?.[key];

          return (
            <tr key={key}>
              <td className="py-2">
                <span className="mr-2 font-semibold text-accent">{symbol}</span>
                <span className="text-ink">{label}</span>
                <span className="ml-2 text-[11px] text-ink-faint">{hint}</span>
              </td>
              <td className="tnum py-2 text-right font-medium text-ink">{fmt(mc)}</td>
              {hasBenchmark && (
                <td className="tnum py-2 text-right text-ink-dim">{fmt(bs)}</td>
              )}
              {hasBenchmark && (
                <td className="tnum py-2 text-right text-ink-faint">
                  {bs === undefined ? "—" : fmt(mc - bs)}
                </td>
              )}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default GreeksTable;
