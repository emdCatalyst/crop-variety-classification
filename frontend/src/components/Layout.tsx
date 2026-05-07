import { useEffect } from "react";
import { Link, NavLink, Outlet, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { logout } from "@/api/auth";
import { User } from "@/api/client";
import LanguageSwitcher from "./LanguageSwitcher";
import NotificationsBell from "./NotificationsBell";

export default function Layout({
  user,
  unread,
  onUnreadChange,
  onSignOut,
  onUserChange,
}: {
  user: User;
  unread: number;
  onUnreadChange: (n: number) => void;
  onSignOut: () => void;
  onUserChange?: (u: User) => void;
}) {
  const navigate = useNavigate();
  const { t, i18n } = useTranslation();

  useEffect(() => {
    if (user.language && user.language !== i18n.language.split("-")[0]) {
      i18n.changeLanguage(user.language);
    }
  }, [user.language, i18n]);

  async function handleSignOut() {
    await logout();
    onSignOut();
    navigate("/login");
  }

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 rounded-md text-sm font-medium ${
      isActive ? "bg-brand-600 text-white" : "text-slate-700 hover:bg-slate-200"
    }`;

  return (
    <div className="min-h-full">
      <nav className="bg-white border-b border-slate-200">
        <div className="container flex items-center justify-between h-14 gap-4">
          <Link to="/" className="font-bold text-brand-700 text-lg whitespace-nowrap">
            {t("app.name")}
          </Link>
          <div className="flex items-center gap-1 flex-wrap">
            <NavLink to="/dashboard" className={linkClass}>
              {t("nav.dashboard")}
            </NavLink>
            <NavLink to="/upload" className={linkClass}>
              {t("nav.upload")}
            </NavLink>
            <NavLink to="/reports" className={linkClass}>
              {t("nav.reports")}
            </NavLink>
            <NavLink to="/settings" className={linkClass}>
              {t("nav.settings")}
            </NavLink>
          </div>
          <div className="flex items-center gap-3">
            <NotificationsBell unread={unread} onUnreadChange={onUnreadChange} />
            <LanguageSwitcher
              authenticated
              onLanguageChange={(lang) => onUserChange?.({ ...user, language: lang })}
            />
            <span className="text-sm text-slate-600 hidden sm:inline">{user.display_name}</span>
            <button
              onClick={handleSignOut}
              className="text-sm text-slate-600 hover:text-brand-700"
            >
              {t("nav.sign_out")}
            </button>
          </div>
        </div>
      </nav>
      <main className="container py-8">
        <Outlet />
      </main>
    </div>
  );
}
