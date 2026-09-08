import { BrowserRouter, Routes, Route } from "react-router-dom";
import TopNav from "./components/TopNav";
import ErrorBoundary from "./components/ErrorBoundary";
import KnowledgeBase from "./pages/KnowledgeBase";
import ChatPage from "./pages/ChatPage";
import DocumentReport from "./pages/DocumentReport";
import DocumentKnowledgeGraph from "./pages/DocumentKnowledgeGraph";
import DocumentView from "./pages/DocumentView";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <ErrorBoundary>
        <div className="app-layout">
          <TopNav />
          <main className="app-main">
            <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/kb" element={<KnowledgeBase />} />
            <Route path="/documents/:id/report" element={<DocumentReport />} />
            <Route path="/documents/:id/kg" element={<DocumentKnowledgeGraph />} />
            <Route path="/documents/:id/view" element={<DocumentView />} />
          </Routes>
        </main>
      </div>
      </ErrorBoundary>
    </BrowserRouter>
  );
}
