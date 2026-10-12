import { Route, Routes } from "react-router-dom";
import PageShell from "./components/layout/PageShell";
import DashboardPage from "./pages/DashboardPage";
import AnalyzePage from "./pages/AnalyzePage";
import HistoryPage from "./pages/HistoryPage";
import PredictionDetailPage from "./pages/PredictionDetailPage";
import AnalyticsPage from "./pages/AnalyticsPage";
import ModelsPage from "./pages/ModelsPage";
import NotFoundPage from "./pages/NotFoundPage";
import ErrorBoundary from "./components/ui/ErrorBoundary";

export default function App() {
  return (
    <PageShell>
      <ErrorBoundary>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/analyze" element={<AnalyzePage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/history/:id" element={<PredictionDetailPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
      </ErrorBoundary>
    </PageShell>
  );
}
