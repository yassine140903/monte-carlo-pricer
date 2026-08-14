import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { AppProvider } from "./context/AppContext";
import CalibratePage from "./pages/CalibratePage";
import RiskPage from "./pages/RiskPage";
import RunsPage from "./pages/RunsPage";
import SimulatePricePage from "./pages/SimulatePricePage";

export function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Navigate to="/calibrate" replace />} />
            <Route path="/calibrate" element={<CalibratePage />} />
            <Route path="/simulate" element={<SimulatePricePage />} />
            <Route path="/risk" element={<RiskPage />} />
            <Route path="/runs" element={<RunsPage />} />
            {/* Anything else lands on the first step of the workflow. */}
            <Route path="*" element={<Navigate to="/calibrate" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AppProvider>
  );
}

export default App;
