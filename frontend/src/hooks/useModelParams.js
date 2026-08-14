import { useCallback, useEffect, useState } from "react";
import { useApp } from "../context/AppContext";
import { defaultParams } from "../lib/models";

/**
 * Round the fitted params to something an input box can show.
 *
 * A calibrated kappa arrives as 11.998750094296934, which overflows the field
 * and cannot be edited without first deleting a screenful of digits. Six
 * decimals is far below the Monte Carlo standard error on anything priced
 * from them — the estimator's own noise is orders of magnitude larger — so
 * this costs nothing real and makes the form usable. The Calibrate page still
 * reports the values it was given.
 */
function forEditing(params) {
  return Object.fromEntries(
    Object.entries(params).map(([key, value]) => [
      key,
      typeof value === "number" ? Number(value.toFixed(6)) : value,
    ]),
  );
}

/**
 * Model params that start from the calibration and stay editable.
 *
 * The simulate, price and risk pages all want the same behaviour: adopt the
 * fitted params when there are any, fall back to hand-set defaults when there
 * are not, and let the user override either.
 *
 * The effect deliberately depends on the *calibrated object* rather than
 * running on every render, so edits survive re-renders and are only discarded
 * when the underlying calibration actually changes — switching model family,
 * or re-fitting the ticker.
 */
export function useModelParams(modelType) {
  const { calibratedParams } = useApp();
  const calibrated = calibratedParams[modelType];

  const [params, setParams] = useState(() => calibrated ? forEditing(calibrated) : defaultParams(modelType));

  useEffect(() => {
    setParams(calibrated ? forEditing(calibrated) : defaultParams(modelType));
  }, [modelType, calibrated]);

  const reset = useCallback(() => {
    setParams(calibrated ? forEditing(calibrated) : defaultParams(modelType));
  }, [modelType, calibrated]);

  return {
    params,
    setParams,
    reset,
    isCalibrated: Boolean(calibrated),
    // model_type must travel with the params: it is the discriminator the API
    // validates the body against.
    body: { ...params, model_type: modelType },
  };
}

export default useModelParams;
