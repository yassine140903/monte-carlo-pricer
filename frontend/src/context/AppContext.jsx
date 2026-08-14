import { createContext, useCallback, useContext, useMemo, useState } from "react";

/**
 * What the pages share: which ticker is in play, what the last calibration
 * fitted for it, and the spot to simulate from.
 *
 * Deliberately small. Everything a single page owns — form inputs, results,
 * loading flags — stays in that page's own state; putting it here would make
 * every keystroke on the pricing form re-render the risk page.
 */
const AppContext = createContext(null);

const EMPTY_PARAMS = { gbm: null, jump_diffusion: null, heston: null };

export function AppProvider({ children }) {
  const [selectedTicker, setSelectedTickerState] = useState(null);
  const [calibratedParams, setCalibratedParams] = useState(EMPTY_PARAMS);
  const [S0, setS0] = useState(100);

  /**
   * Changing ticker clears the fitted params. They describe the *old* ticker,
   * and silently pricing one company's option off another's volatility is the
   * kind of wrong that looks perfectly plausible on screen.
   */
  const setSelectedTicker = useCallback((ticker) => {
    setSelectedTickerState((current) => {
      if (current !== ticker) setCalibratedParams(EMPTY_PARAMS);
      return ticker;
    });
  }, []);

  const storeCalibration = useCallback((modelType, params) => {
    setCalibratedParams((current) => ({ ...current, [modelType]: params }));
  }, []);

  const clearCalibration = useCallback(() => setCalibratedParams(EMPTY_PARAMS), []);

  const value = useMemo(
    () => ({
      selectedTicker,
      setSelectedTicker,
      calibratedParams,
      storeCalibration,
      clearCalibration,
      hasCalibration: Object.values(calibratedParams).some(Boolean),
      S0,
      setS0,
    }),
    [selectedTicker, setSelectedTicker, calibratedParams, storeCalibration, clearCalibration, S0],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used inside <AppProvider>");
  return context;
}

export default AppContext;
