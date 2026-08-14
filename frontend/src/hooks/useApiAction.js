import { useCallback, useRef, useState } from "react";
import { apiErrorMessage } from "../api/client";

/**
 * The loading / data / error triple every endpoint call needs, in one place.
 *
 * Two things it handles that a bare `await` in a component does not:
 *
 * - Stale responses. A 50,000-path simulation can still be in flight when the
 *   user fires a smaller one; without the request counter the slow reply would
 *   land second and overwrite the result actually being looked at.
 * - Unmounting. Switching tabs mid-request would otherwise set state on a
 *   component that is gone.
 */
export function useApiAction(requestFn) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const latestRequest = useRef(0);

  const run = useCallback(
    async (...args) => {
      const ticket = ++latestRequest.current;
      setLoading(true);
      setError(null);

      try {
        const response = await requestFn(...args);
        if (ticket !== latestRequest.current) return null;
        setData(response.data);
        return response.data;
      } catch (err) {
        if (ticket !== latestRequest.current) return null;
        setError(apiErrorMessage(err));
        setData(null);
        return null;
      } finally {
        if (ticket === latestRequest.current) setLoading(false);
      }
    },
    [requestFn],
  );

  const reset = useCallback(() => {
    latestRequest.current += 1;
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, error, loading, run, reset };
}

export default useApiAction;
