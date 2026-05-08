import { useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { fetchMe } from "./api/auth";
import { User } from "./api/client";
import Layout from "./components/Layout";
import HomePage from "./pages/Home";
import LoginPage from "./pages/Login";
import SignupPage from "./pages/Signup";
import DashboardPage from "./pages/Dashboard";
import UploadPage from "./pages/Upload";
import ResultPage from "./pages/Result";
import ReportsPage from "./pages/Reports";
import SettingsPage from "./pages/Settings";
import NotificationsPage from "./pages/Notifications";
import MessagesPage from "./pages/Messages";
import AdminDashboardPage from "./pages/admin/AdminDashboard";
import AdminUsersPage from "./pages/admin/AdminUsers";
import AdminAnalysesPage from "./pages/admin/AdminAnalyses";
import AdminBroadcastPage from "./pages/admin/AdminBroadcast";

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const [unread, setUnread] = useState(0);
  const [unreadMessages, setUnreadMessages] = useState(0);
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
        <Route path="/" element={<HomePage />} />
        <Route path="/login" element={<LoginPage onSignIn={setUser} />} />
        <Route path="/signup" element={<SignupPage onSignIn={setUser} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    );
  }

  const isAdmin = user.role === "admin";

  return (
    <Routes>
      <Route
        element={
          <Layout
            user={user}
            unread={unread}
            unreadMessages={unreadMessages}
            onUnreadChange={setUnread}
            onUnreadMessagesChange={setUnreadMessages}
            onSignOut={() => {
              setUser(null);
              setUnread(0);
              setUnreadMessages(0);
            }}
            onUserChange={setUser}
          />
        }
      >
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route
          path="/dashboard"
          element={isAdmin ? <AdminDashboardPage /> : <DashboardPage />}
        />
        {isAdmin ? (
          <>
            <Route path="/users" element={<AdminUsersPage currentUser={user} />} />
            <Route path="/analyses" element={<AdminAnalysesPage />} />
            <Route path="/broadcast" element={<AdminBroadcastPage />} />
          </>
        ) : (
          <>
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/reports" element={<ReportsPage />} />
          </>
        )}
        <Route path="/analyses/:id" element={<ResultPage />} />
        <Route path="/settings" element={<SettingsPage user={user} onUserChange={setUser} />} />
        <Route
          path="/notifications"
          element={<NotificationsPage onUnreadChange={setUnread} />}
        />
        <Route
          path="/messages"
          element={<MessagesPage user={user} onUnreadChange={setUnreadMessages} />}
        />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  );
}
