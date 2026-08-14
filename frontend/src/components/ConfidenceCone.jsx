import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtMoney } from "../lib/format";

/**
 * The signature chart: a fan of percentile bands with a few individual paths
 * drawn through it.
 *
 * Each band is a *range* series — the [low, high] pairs built in
 * useSimulation's toChartData — so it floats between two percentiles instead
 * of filling up from the axis. Two nested bands are drawn, 5–95 then 25–75
 * over the top of it, which puts the heaviest ink where the paths actually
 * concentrate and leaves the tails visibly fainter.
 *
 * Sample paths go on last, thin and semi-transparent — they show the texture
 * the bands smooth away without competing with them.
 */

const PERCENTILE_ROWS = [
  { key: "p95", label: "95th" },
  { key: "p75", label: "75th" },
  { key: "p50", label: "Median" },
  { key: "p25", label: "25th" },
  { key: "p5", label: "5th" },
];

function ConeTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;

  return (
    <div className="rounded-lg border border-edge bg-surface px-3 py-2 shadow-xl">
      <p className="tnum mb-1.5 text-[11px] text-ink-dim">t = {label.toFixed(3)} y</p>
      <table className="tnum text-xs">
        <tbody>
          {PERCENTILE_ROWS.map(({ key, label: name }) => (
            <tr key={key}>
              <td className="pr-3 text-ink-faint">{name}</td>
              <td
                className={`text-right font-medium ${
                  key === "p50" ? "text-accent" : "text-ink"
                }`}
              >
                {fmtMoney(row[key])}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ConfidenceCone({ chart, height = 380 }) {
  if (!chart) return null;

  const [low, high] = chart.yDomain;
  const pad = (high - low) * 0.04;

  return (
    <div className="w-full">
      {/* Only the plot is height-constrained — putting the legend inside this
          box too would push it out under whatever follows the chart. */}
      <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chart.cone} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <defs>
            <linearGradient id="band-outer" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--color-band-outer)" stopOpacity={0.42} />
              <stop offset="100%" stopColor="var(--color-band-outer)" stopOpacity={0.18} />
            </linearGradient>
            <linearGradient id="band-mid" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--color-band-core)" stopOpacity={0.55} />
              <stop offset="100%" stopColor="var(--color-band-mid)" stopOpacity={0.35} />
            </linearGradient>
          </defs>

          <CartesianGrid stroke="#ffffff" strokeOpacity={0.05} vertical={false} />

          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(t) => `${t.toFixed(2)}y`}
            stroke="var(--color-ink-faint)"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "var(--color-edge)" }}
          />
          <YAxis
            domain={[low - pad, high + pad]}
            tickFormatter={(v) => fmtMoney(v, 0)}
            stroke="var(--color-ink-faint)"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={58}
          />

          <Tooltip content={<ConeTooltip />} cursor={{ stroke: "var(--color-edge)" }} />

          <Area
            dataKey="outer"
            stroke="none"
            fill="url(#band-outer)"
            isAnimationActive={false}
          />
          <Area
            dataKey="inner"
            stroke="none"
            fill="url(#band-mid)"
            isAnimationActive={false}
          />

          {chart.pathKeys.map((key) => (
            <Line
              key={key}
              dataKey={key}
              stroke="var(--color-ink)"
              strokeOpacity={0.32}
              strokeWidth={1}
              dot={false}
              isAnimationActive={false}
              legendType="none"
            />
          ))}

          <Line
            dataKey="p50"
            stroke="var(--color-accent)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1.5 px-1 text-[11px] text-ink-faint">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm bg-[var(--color-band-core)] opacity-60" />
          25–75th percentile
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm bg-[var(--color-band-outer)] opacity-45" />
          5–95th percentile
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded bg-accent" />
          median
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded bg-ink opacity-40" />
          sample paths
        </span>
      </div>
    </div>
  );
}

export default ConfidenceCone;
