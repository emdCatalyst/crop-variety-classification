import { useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { fetchMe } from "./api/auth";
import { User } from "./api/client";
import Layout from "./components/Layout";
import LoginPage from "./pages/Login";
import SignupPage from "./pages/Signup";
import DashboardPage from "./pages/Dashboard";
import UploadPage from "./pages/Upload";
import ResultPage from "./pages/Result";
import ReportsPage from "./pages/Reports";
import SettingsPage from "./pages/Settings";
import NotificationsPage from "./pages/Notifications";

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const [unread, setUnread] = useState(0);
  const { t } = useTranslation();

  useEffect(() => {
    fetchMe().then((u) => {
      setUser(u);
      setReady(true);
    });
  }, []);

  if (!ready) {
    return (
      <div className="min-h-screen flex items-center justify-center text-slate-500">
        {t("common.loading")}
      </div>
    );
  }

  if (!user) {
    return (
      <Routes>
        <Route path="/login" element={<LoginPage onSignIn={setUser} />} />
        <Route path="/signup" element={<SignupPage onSignIn={setUser} />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }

  return (
    <Routes>
      <Route
        element={
          <Layout
            user={user}
            unread={unread}
            onUnreadChange={setUnread}
            onSignOut={() => {
              setUser(null);
              setUnread(0);
            }}
            onUserChange={setUser}
          />
        }
      >
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/analyses/:id" element={<ResultPage />} />
        <Route path="/reports" element={<ReportsPage />} />
        <Route path="/settings" element={<SettingsPage user={user} onUserChange={setUser} />} />
        <Route
          path="/notifications"
          element={<NotificationsPage onUnreadChange={setUnread} />}
        />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  );
}
