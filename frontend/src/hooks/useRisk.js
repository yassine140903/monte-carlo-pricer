import { useCallback, useEffect, useState } from "react";
import { apiErrorMessage, getScenarios, portfolioRisk, riskMetrics } from "../api/client";
import useApiAction from "./useApiAction";

/** Single-asset risk, plus the stressed run when a scenario is selected. */
export function useSingleAssetRisk() {
  const [baseline, setBaseline] = useState(null);
  const [stressed, setStressed] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const run = useCallback(async (body, scenarioKey) => {
    setLoading(true);
    setError(null);

    try {
      // Both runs share the request's seed, so the difference between them is
      // the shock and not two independent samples. The scenario is applied
      // server-side — apply_scenario stays the one definition of what a
      // scenario means to a params object.
      const calls = [riskMetrics(body)];
      if (scenarioKey) calls.push(riskMetrics({ ...body, scenario: scenarioKey }));

      const [base, stress] = await Promise.all(calls);
      setBaseline(base.data);
      setStressed(stress?.data ?? null);
      return base.data;
    } catch (err) {
      setError(apiErrorMessage(err));
      setBaseline(null);
      setStressed(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setBaseline(null);
    setStressed(null);
    setError(null);
  }, []);

  return { baseline, stressed, error, loading, run, reset };
}

export function usePortfolioRisk() {
  return useApiAction(portfolioRisk);
}

/**
 * The preset stress scenarios, fetched once.
 *
 * Each family lists the model families it has a variant for — a scenario names
 * parameters, and `v0` means nothing to GBM — so the UI can offer only the
 * ones that apply to the model currently selected.
 */
export function useScenarios() {
  const [scenarios, setScenarios] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    getScenarios()
      .then((response) => {
        if (active) setScenarios(response.data.scenarios);
      })
      .catch((err) => {
        if (active) setError(apiErrorMessage(err));
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const forModel = useCallback(
    (modelType) => scenarios.filter((family) => family.model_types.includes(modelType)),
    [scenarios],
  );

  return { scenarios, forModel, error, loading };
}

export default useSingleAssetRisk;
