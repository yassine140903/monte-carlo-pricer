import { useCallback, useState } from "react";
import { apiErrorMessage, calibrate } from "../api/client";
import { useApp } from "../context/AppContext";
import { MODEL_KEYS } from "../lib/models";

const idle = () => ({ status: "idle", data: null, error: null });

/**
 * Fits all three model families to one ticker at once.
 *
 * The three calls are independent, so they go out in parallel and are settled
 * individually: Heston's term-structure fit can fail on a short history while
 * GBM succeeds, and the page should show the two that worked rather than one
 * error for everything. Whatever succeeds is pushed into AppContext so the
 * simulate, price and risk pages can use it without re-fitting.
 */
export function useCalibration() {
  const { storeCalibration, setSelectedTicker } = useApp();
  const [results, setResults] = useState(() => ({
    gbm: idle(),
    jump_diffusion: idle(),
    heston: idle(),
  }));
  const [loading, setLoading] = useState(false);

  const run = useCallback(
    async (ticker, { lookbackDays, windowDays } = {}) => {
      setLoading(true);
      setSelectedTicker(ticker);
      setResults({
        gbm: { status: "loading", data: null, error: null },
        jump_diffusion: { status: "loading", data: null, error: null },
        heston: { status: "loading", data: null, error: null },
      });

      const settled = await Promise.allSettled(
        MODEL_KEYS.map((modelType) =>
          calibrate({
            ticker,
            model_type: modelType,
            lookback_days: lookbackDays,
            // Only GBM reads window_days; the API ignores it elsewhere.
            window_days: modelType === "gbm" ? (windowDays ?? null) : null,
          }),
        ),
      );

      const next = {};
      settled.forEach((outcome, index) => {
        const modelType = MODEL_KEYS[index];
        if (outcome.status === "fulfilled") {
          next[modelType] = { status: "success", data: outcome.value.data, error: null };
          storeCalibration(modelType, outcome.value.data.params);
        } else {
          next[modelType] = {
            status: "error",
            data: null,
            error: apiErrorMessage(outcome.reason),
          };
        }
      });

      setResults(next);
      setLoading(false);
      return next;
    },
    [storeCalibration, setSelectedTicker],
  );

  const reset = useCallback(() => {
    setResults({ gbm: idle(), jump_diffusion: idle(), heston: idle() });
  }, []);

  return { results, loading, run, reset };
}

export default useCalibration;
