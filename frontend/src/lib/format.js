/** Number formatting for readouts. Charts do their own axis formatting. */

export function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  const abs = Math.abs(value);
  // Below 1e-4 a fixed-decimal render collapses to 0.0000, which reads as
  // "zero" rather than "very small". Exponential keeps the magnitude visible.
  if (abs !== 0 && (abs < 1e-4 || abs >= 1e7)) return value.toExponential(2);
  return value.toFixed(digits);
}

export function fmtMoney(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtPercent(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function fmtCount(value) {
  if (value === null || value === undefined) return "—";
  return value.toLocaleString();
}

export function fmtMillis(ms) {
  if (ms === null || ms === undefined) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}
