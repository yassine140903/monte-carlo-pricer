import { useMemo } from "react";
import { simulate } from "../api/client";
import useApiAction from "./useApiAction";

/**
 * Reshapes /simulate's column-wise response into the row-wise form Recharts
 * wants.
 *
 * The API sends `{time_axis: [...], percentile_bands: {p5: [...], ...}}` —
 * parallel arrays, already downsampled to the same ~50 dates. Recharts needs
 * one object per x position, so the arrays are zipped once here rather than
 * inside the chart's render.
 *
 * The bands become *ranges* — a [low, high] pair per date, which Recharts
 * draws as a band floating between the two rather than as a fill rising from
 * the axis. The alternative, stacking a hidden pedestal under four slabs,
 * draws the same picture but drags the stack's zero baseline into the y-axis
 * domain, which flattens the whole cone against the top of the plot.
 */
function toChartData(response) {
  if (!response) return null;

  const { time_axis: t, percentile_bands: bands, sample_paths: paths } = response;

  const cone = t.map((time, i) => ({
    t: time,
    outer: [bands.p5[i], bands.p95[i]],
    inner: [bands.p25[i], bands.p75[i]],
    p5: bands.p5[i],
    p25: bands.p25[i],
    p50: bands.p50[i],
    p75: bands.p75[i],
    p95: bands.p95[i],
    ...Object.fromEntries(paths.map((path, k) => [`path${k}`, path[i]])),
  }));

  const values = [...bands.p5, ...bands.p95, ...paths.flat()];

  return {
    cone,
    pathKeys: paths.map((_, k) => `path${k}`),
    yDomain: [Math.min(...values), Math.max(...values)],
  };
}

export function useSimulation() {
  const action = useApiAction(simulate);
  const chart = useMemo(() => toChartData(action.data), [action.data]);

  return { ...action, chart };
}

export default useSimulation;
