import axios from "axios";

export const API_BASE_URL = "http://localhost:8000";
export const MLFLOW_UI_URL = "http://localhost:5000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});

/**
 * The backend answers every failure with {error, detail, context} (see
 * src/api/main.py). Unwrap that into a plain Error so callers can render
 * `err.message` without knowing the envelope — and so a network failure,
 * which has no envelope at all, still produces something readable.
 */
export function apiErrorMessage(error) {
  const payload = error?.response?.data;

  if (payload?.detail) {
    const rows = payload.context?.errors;
    if (Array.isArray(rows) && rows.length > 0) {
      return rows
        .map((row) => `${row.loc?.filter((p) => p !== "body").join(".")}: ${row.msg}`)
        .join("; ");
    }
    return payload.detail;
  }

  if (error?.response) {
    return `Request failed (HTTP ${error.response.status})`;
  }
  return `Cannot reach the API at ${API_BASE_URL} — is the backend running?`;
}

// --- Data ---------------------------------------------------------------

export const getTickers = () => api.get("/data/tickers");

// The query params are start_date/end_date, matching the FastAPI signature.
export const getTickerData = (ticker, startDate, endDate) =>
  api.get(`/data/${ticker}`, {
    params: { start_date: startDate, end_date: endDate },
  });

// --- Calibration --------------------------------------------------------

export const calibrate = (body) => api.post("/calibrate", body);

// --- Simulation ---------------------------------------------------------

export const simulate = (body) => api.post("/simulate", body);

// --- Pricing ------------------------------------------------------------

export const priceOption = (body) => api.post("/price-option", body);

// --- Risk ---------------------------------------------------------------

export const riskMetrics = (body) => api.post("/risk/metrics", body);
export const portfolioRisk = (body) => api.post("/risk/portfolio", body);
export const getScenarios = () => api.get("/risk/scenarios");

// --- MLflow -------------------------------------------------------------

export const getMlflowExperiments = () => api.get("/mlflow/experiments");

export const getHealth = () => api.get("/health");

export default api;
