import { AlertCircle } from "lucide-react";

/**
 * The small primitives every panel is built from — card chrome, form controls,
 * the error banner. Kept together so the dark-theme classes are written once;
 * the feature components below import from here rather than repeating them.
 */

export function Card({ title, subtitle, right, children, className = "" }) {
  return (
    <section
      className={`rounded-xl border border-edge bg-surface shadow-lg shadow-black/20 ${className}`}
    >
      {(title || right) && (
        <header className="flex items-start justify-between gap-4 border-b border-edge px-5 py-3.5">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-dim">{subtitle}</p>}
          </div>
          {right && <div className="shrink-0">{right}</div>}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Field({ label, hint, children }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-ink-dim">
        {label}
        {hint && <span className="ml-1.5 font-normal text-ink-faint">{hint}</span>}
      </span>
      {children}
    </label>
  );
}

const controlClasses =
  "w-full rounded-lg border border-edge bg-surface-2 px-3 py-2 text-sm text-ink " +
  "outline-none transition focus:border-accent focus:ring-1 focus:ring-accent " +
  "disabled:cursor-not-allowed disabled:opacity-50";

export function NumberInput({ value, onChange, ...props }) {
  return (
    <input
      type="number"
      className={`${controlClasses} tnum`}
      value={value}
      onChange={(e) => {
        const raw = e.target.value;
        // An empty box is a legitimate mid-typing state; coercing it to 0
        // would fight the user as they clear a field to retype it.
        onChange(raw === "" ? "" : Number(raw));
      }}
      {...props}
    />
  );
}

export function Select({ value, onChange, options, ...props }) {
  return (
    <select
      className={controlClasses}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      {...props}
    >
      {options.map((option) => (
        <option key={option.key} value={option.key}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

export function Button({ children, variant = "primary", className = "", ...props }) {
  const variants = {
    primary:
      "bg-accent text-base font-semibold hover:bg-accent-dim disabled:hover:bg-accent",
    ghost:
      "border border-edge bg-surface-2 text-ink hover:border-accent hover:text-accent",
  };
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm
        transition disabled:cursor-not-allowed disabled:opacity-50 ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function Toggle({ value, onChange, options }) {
  return (
    <div className="inline-flex rounded-lg border border-edge bg-surface-2 p-1">
      {options.map((option) => (
        <button
          key={option.key}
          type="button"
          onClick={() => onChange(option.key)}
          className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
            value === option.key
              ? "bg-accent text-base"
              : "text-ink-dim hover:text-ink"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Badge({ children, tone = "neutral" }) {
  const tones = {
    neutral: "border-edge bg-surface-2 text-ink-dim",
    good: "border-good/40 bg-good/10 text-good",
    warn: "border-warn/40 bg-warn/10 text-warn",
    bad: "border-bad/40 bg-bad/10 text-bad",
    accent: "border-accent/40 bg-accent/10 text-accent",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px]
        font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="flex items-start gap-2 rounded-lg border border-bad/40 bg-bad/10 px-3 py-2.5">
      <AlertCircle size={15} className="mt-0.5 shrink-0 text-bad" />
      <p className="text-xs leading-relaxed text-bad">{message}</p>
    </div>
  );
}

export function EmptyState({ children }) {
  return (
    <div className="flex min-h-32 items-center justify-center rounded-lg border border-dashed border-edge px-6 py-10">
      <p className="max-w-md text-center text-sm text-ink-faint">{children}</p>
    </div>
  );
}

/** A labelled number, the unit of every summary strip in the app. */
export function Stat({ label, value, tone = "neutral", hint }) {
  const tones = {
    neutral: "text-ink",
    good: "text-good",
    bad: "text-bad",
    accent: "text-accent",
  };
  return (
    <div className="rounded-lg border border-edge bg-surface-2 px-3.5 py-3">
      <p className="text-[11px] font-medium tracking-wide text-ink-dim uppercase">{label}</p>
      <p className={`tnum mt-1 text-lg font-semibold ${tones[tone]}`}>{value}</p>
      {hint && <p className="mt-0.5 text-[11px] text-ink-faint">{hint}</p>}
    </div>
  );
}
