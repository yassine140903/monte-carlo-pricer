import { useMemo } from "react";
import { priceOption } from "../api/client";
import useApiAction from "./useApiAction";

/**
 * Turns the API's `{bin_edges, counts}` histogram into Recharts bars.
 *
 * `bin_edges` has one more entry than `counts` — the NumPy convention — so bar
 * i spans edges[i]..edges[i+1]. The bar is positioned at the bin *midpoint*
 * and carries its range as a label, which keeps the x-axis a real numeric
 * scale instead of 50 evenly spaced category strings.
 */
function toHistogramData(histogram) {
  if (!histogram) return null;

  const { bin_edges: edges, counts } = histogram;
  const decimals = Math.abs(edges[edges.length - 1] - edges[0]) < 10 ? 2 : 1;

  return counts.map((count, i) => ({
    mid: (edges[i] + edges[i + 1]) / 2,
    range: `${edges[i].toFixed(decimals)} – ${edges[i + 1].toFixed(decimals)}`,
    count,
  }));
}

export function usePricing() {
  const action = useApiAction(priceOption);

  const histogram = useMemo(
    () => toHistogramData(action.data?.payoff_histogram),
    [action.data],
  );

  return { ...action, histogram };
}

export default usePricing;
