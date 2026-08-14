import { Loader2 } from "lucide-react";

export function LoadingSpinner({ label, size = 16, className = "" }) {
  return (
    <span className={`inline-flex items-center gap-2 text-ink-dim ${className}`}>
      <Loader2 size={size} className="animate-spin text-accent" />
      {label && <span className="text-xs">{label}</span>}
    </span>
  );
}

/** Fills a panel while its contents are in flight, so the layout holds still. */
export function LoadingPanel({ label = "Working…", minHeight = "min-h-64" }) {
  return (
    <div className={`flex ${minHeight} items-center justify-center`}>
      <LoadingSpinner label={label} size={22} />
    </div>
  );
}

export default LoadingSpinner;
