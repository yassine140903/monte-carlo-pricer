import { Activity, FlaskConical, LineChart, ShieldAlert, Sigma } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { useApp } from "../context/AppContext";

const TABS = [
  { to: "/calibrate", label: "Calibrate", Icon: Sigma },
  { to: "/simulate", label: "Simulate & Price", Icon: LineChart },
  { to: "/risk", label: "Risk", Icon: ShieldAlert },
  { to: "/runs", label: "Runs", Icon: FlaskConical },
];

export function Layout() {
  const { selectedTicker, hasCalibration } = useApp();

  return (
    <div className="min-h-full bg-base">
      <header className="sticky top-0 z-10 border-b border-edge bg-base/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-8 gap-y-3 px-6 py-3">
          <div className="flex items-center gap-2.5">
            <Activity size={20} className="text-accent" />
            <span className="text-sm font-semibold tracking-tight">Monte Carlo Pricer</span>
          </div>

          <nav className="flex items-center gap-1">
            {TABS.map(({ to, label, Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm transition ${
                    isActive
                      ? "bg-surface-2 font-medium text-accent"
                      : "text-ink-dim hover:bg-surface hover:text-ink"
                  }`
                }
              >
                <Icon size={15} />
                {label}
              </NavLink>
            ))}
          </nav>

          {/* The active ticker drives every other page, so it stays visible. */}
          <div className="ml-auto flex items-center gap-2 text-xs">
            {selectedTicker ? (
              <>
                <span className="text-ink-faint">Ticker</span>
                <span className="rounded-md border border-edge bg-surface-2 px-2 py-1 font-semibold text-ink">
                  {selectedTicker}
                </span>
                {hasCalibration && (
                  <span className="rounded-md border border-good/40 bg-good/10 px-2 py-1 font-medium text-good">
                    calibrated
                  </span>
                )}
              </>
            ) : (
              <span className="text-ink-faint">No ticker selected</span>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-6 py-6">
        <Outlet />
      </main>
    </div>
  );
}

export default Layout;
