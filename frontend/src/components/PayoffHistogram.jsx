import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtCount, fmtMoney } from "../lib/format";

/**
 * Distribution of the undiscounted payoff across simulated paths.
 *
 * The API has already bucketed it — 50 bins from np.histogram — so this only
 * draws what it is given. The x-axis is the bin midpoint on a numeric scale
 * rather than 50 category labels, which keeps the spacing honest when the bins
 * are uneven and lets the mass at zero (every path that expired worthless)
 * read as a spike rather than as one bar among fifty.
 */

function HistogramTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;

  return (
    <div className="rounded-lg border border-edge bg-surface px-3 py-2 shadow-xl">
      <p className="tnum text-[11px] text-ink-dim">payoff {row.range}</p>
      <p className="tnum mt-0.5 text-sm font-medium text-accent">
        {fmtCount(row.count)} paths
      </p>
    </div>
  );
}

export function PayoffHistogram({ data, height = 240 }) {
  if (!data?.length) return null;

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <defs>
            <linearGradient id="payoff-bar" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--color-accent)" stopOpacity={0.9} />
              <stop offset="100%" stopColor="var(--color-accent-dim)" stopOpacity={0.45} />
            </linearGradient>
          </defs>

          <CartesianGrid stroke="#ffffff" strokeOpacity={0.05} vertical={false} />

          <XAxis
            dataKey="mid"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v) => fmtMoney(v, 0)}
            stroke="var(--color-ink-faint)"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "var(--color-edge)" }}
          />
          <YAxis
            stroke="var(--color-ink-faint)"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={52}
            tickFormatter={fmtCount}
          />

          <Tooltip
            content={<HistogramTooltip />}
            cursor={{ fill: "var(--color-surface-2)", fillOpacity: 0.5 }}
          />

          <Bar dataKey="count" fill="url(#payoff-bar)" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default PayoffHistogram;
