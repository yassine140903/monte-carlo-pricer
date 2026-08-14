import { useEffect, useState } from "react";
import { apiErrorMessage, getTickers } from "../api/client";
import { ErrorBanner, Field } from "./ui";
import { LoadingSpinner } from "./LoadingSpinner";

/**
 * Dropdown over whatever is actually in the database.
 *
 * The list is deliberately not hardcoded from config.SUPPORTED_TICKERS: that
 * is the set the fetcher *may* ingest, while /data/tickers is the set that has
 * actually been ingested. Offering a ticker with no stored bars would only
 * produce a 404 one click later.
 */
export function TickerSelector({ value, onChange, onMeta, disabled }) {
  const [tickers, setTickers] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    getTickers()
      .then((response) => {
        if (!active) return;
        setTickers(response.data);
        // Preselect the first one so the page is usable in a single click.
        if (!value && response.data.length > 0) onChange(response.data[0].ticker);
      })
      .catch((err) => active && setError(apiErrorMessage(err)))
      .finally(() => active && setLoading(false));

    return () => {
      active = false;
    };
    // Runs once: refetching whenever the selection changes would be pointless
    // work, and would fight the preselect above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selected = tickers.find((t) => t.ticker === value);

  // Hand the selected ticker's date range up, so the page can go fetch its
  // last close without this component having to know why it wants it.
  useEffect(() => {
    if (selected && onMeta) onMeta(selected);
  }, [selected, onMeta]);

  if (error) return <ErrorBanner message={error} />;

  return (
    <div className="flex flex-col gap-1.5">
      <Field label="Ticker">
        <select
          className="w-full rounded-lg border border-edge bg-surface-2 px-3 py-2 text-sm
            text-ink outline-none transition focus:border-accent focus:ring-1
            focus:ring-accent disabled:cursor-not-allowed disabled:opacity-50"
          value={value ?? ""}
          disabled={disabled || loading || tickers.length === 0}
          onChange={(e) => onChange(e.target.value)}
        >
          {tickers.length === 0 && <option value="">No stored tickers</option>}
          {tickers.map((entry) => (
            <option key={entry.ticker} value={entry.ticker}>
              {entry.ticker}
            </option>
          ))}
        </select>
      </Field>

      {loading && <LoadingSpinner label="Loading tickers…" />}

      {selected && (
        <p className="tnum text-[11px] text-ink-faint">
          {selected.row_count.toLocaleString()} bars · {selected.earliest_date} →{" "}
          {selected.latest_date}
        </p>
      )}

      {!loading && tickers.length === 0 && (
        <p className="text-[11px] text-warn">
          The database is empty — ingest some history with src/data/fetcher.py first.
        </p>
      )}
    </div>
  );
}

export default TickerSelector;
