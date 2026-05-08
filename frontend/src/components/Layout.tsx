import { useCallback, useEffect, useRef, useState } from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { logout } from "@/api/auth";
import { User } from "@/api/client";
import { unreadCount as fetchUnreadMessages } from "@/api/messages";
import { playNotificationChime } from "@/lib/chime";
import LanguageSwitcher from "./LanguageSwitcher";
import NotificationsBell from "./NotificationsBell";

const MESSAGES_STREAM_URL = "/api/v1/messages/stream";

export default function Layout({
  user,
  unread,
  unreadMessages,
  onUnreadChange,
  onUnreadMessagesChange,
  onSignOut,
  onUserChange,
}: {
  user: User;
  unread: number;
  unreadMessages: number;
  onUnreadChange: (n: number) => void;
  onUnreadMessagesChange: (n: number) => void;
  onSignOut: () => void;
  onUserChange?: (u: User) => void;
}) {
  const navigate = useNavigate();
  const location = useLocation();
  const { t, i18n } = useTranslation();
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    if (user.language && user.language !== i18n.language.split("-")[0]) {
      i18n.changeLanguage(user.language);
    }
  }, [user.language, i18n]);

  const lastMessagesRef = useRef(unreadMessages);
  useEffect(() => {
    lastMessagesRef.current = unreadMessages;
  }, [unreadMessages]);

  const refreshMessageCount = useCallback(
    async (playOnIncrease: boolean) => {
      try {
        const n = await fetchUnreadMessages();
        if (playOnIncrease && n > lastMessagesRef.current) {
          playNotificationChime();
        }
        onUnreadMessagesChange(n);
      } catch {
        // best-effort
      }
    },
    [onUnreadMessagesChange]
  );

  useEffect(() => {
    refreshMessageCount(false);
    const pollId = window.setInterval(() => refreshMessageCount(true), 60_000);
    return () => window.clearInterval(pollId);
  }, [refreshMessageCount]);

  useEffect(() => {
    let es: EventSource | null = null;
    try {
      es = new EventSource(MESSAGES_STREAM_URL, { withCredentials: true });
      es.addEventListener("message", () => refreshMessageCount(true));
      es.onerror = () => {
        // browser auto-retries
      };
    } catch {
      // ignore
    }
    return () => es?.close();
  }, [refreshMessageCount]);

  async function handleSignOut() {
    await logout();
    onSignOut();
    navigate("/login");
  }

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 rounded-md text-sm font-medium ${
      isActive ? "bg-brand-600 text-white" : "text-slate-700 hover:bg-slate-200"
    }`;

  const messagesBadge =
    unreadMessages > 0 ? (
      <span className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full bg-red-600 text-white text-[10px] font-semibold">
        {unreadMessages > 99 ? "99+" : unreadMessages}
      </span>
    ) : null;

  return (
    <div className="min-h-full">
      <nav className="bg-white border-b border-slate-200">
        <div className="container flex items-center justify-between h-14 gap-2">
          <Link to="/" className="font-bold text-brand-700 text-lg whitespace-nowrap">
            {t("app.name")}
          </Link>

          <div className="hidden md:flex items-center gap-1 flex-1 justify-center min-w-0">
            <NavLink to="/dashboard" className={linkClass}>
              {t("nav.dashboard")}
            </NavLink>
            {user.role === "admin" ? (
              <>
                <NavLink to="/users" className={linkClass}>
                  {t("nav.users")}
                </NavLink>
                <NavLink to="/analyses" className={linkClass}>
                  {t("nav.analyses")}
                </NavLink>
                <NavLink to="/broadcast" className={linkClass}>
                  {t("nav.broadcast")}
                </NavLink>
              </>
            ) : (
              <>
                <NavLink to="/upload" className={linkClass}>
                  {t("nav.upload")}
                </NavLink>
                <NavLink to="/reports" className={linkClass}>
                  {t("nav.reports")}
                </NavLink>
              </>
            )}
            <NavLink to="/messages" className={linkClass}>
              <span className="inline-flex items-center gap-1.5">
                {t("nav.messages")}
                {messagesBadge}
              </span>
            </NavLink>
            <NavLink to="/settings" className={linkClass}>
              {t("nav.settings")}
            </NavLink>
          </div>

          <div className="flex items-center gap-2 md:gap-3">
            <NotificationsBell unread={unread} onUnreadChange={onUnreadChange} />
            <LanguageSwitcher
              authenticated
              onLanguageChange={(lang) => onUserChange?.({ ...user, language: lang })}
            />
            <span className="text-sm text-slate-600 hidden lg:inline">
              {user.display_name}
            </span>
            <button
              onClick={handleSignOut}
              className="hidden md:inline text-sm text-slate-600 hover:text-brand-700"
            >
              {t("nav.sign_out")}
            </button>
            <button
              type="button"
              onClick={() => setMenuOpen((v) => !v)}
              className="md:hidden relative inline-flex items-center justify-center w-9 h-9 rounded-md text-slate-700 hover:bg-slate-100"
              aria-label={t("nav.menu") ?? "Menu"}
              aria-expanded={menuOpen}
            >
              {menuOpen ? <CloseIcon /> : <BurgerIcon />}
              {!menuOpen && unreadMessages > 0 && (
                <span className="absolute -top-1 -end-1 inline-flex items-center justify-center min-w-[16px] h-[16px] px-1 rounded-full bg-red-600 text-white text-[9px] font-semibold">
                  {unreadMessages > 9 ? "9+" : unreadMessages}
                </span>
              )}
            </button>
          </div>
        </div>

        {menuOpen && (
          <div className="md:hidden border-t border-slate-200 bg-white">
            <div className="container py-2 flex flex-col gap-1">
              <MobileLink to="/dashboard">{t("nav.dashboard")}</MobileLink>
              {user.role === "admin" ? (
                <>
                  <MobileLink to="/users">{t("nav.users")}</MobileLink>
                  <MobileLink to="/analyses">{t("nav.analyses")}</MobileLink>
                  <MobileLink to="/broadcast">{t("nav.broadcast")}</MobileLink>
                </>
              ) : (
                <>
                  <MobileLink to="/upload">{t("nav.upload")}</MobileLink>
                  <MobileLink to="/reports">{t("nav.reports")}</MobileLink>
                </>
              )}
              <MobileLink to="/messages">
                <span className="inline-flex items-center gap-1.5">
                  {t("nav.messages")}
                  {messagesBadge}
                </span>
              </MobileLink>
              <MobileLink to="/settings">{t("nav.settings")}</MobileLink>
              <div className="border-t border-slate-200 my-1" />
              <div className="px-3 py-1.5 text-xs text-slate-500">
                {user.display_name}
              </div>
              <button
                onClick={handleSignOut}
                className="text-start px-3 py-2 rounded-md text-sm text-slate-700 hover:bg-slate-100"
              >
                {t("nav.sign_out")}
              </button>
            </div>
          </div>
        )}
      </nav>
      <main className="container py-8">
        <Outlet />
      </main>
    </div>
  );
}

function MobileLink({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-3 py-2 rounded-md text-sm font-medium ${
          isActive ? "bg-brand-600 text-white" : "text-slate-700 hover:bg-slate-100"
        }`
      }
    >
      {children}
    </NavLink>
  );
}

function BurgerIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      aria-hidden
    >
      <path d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      aria-hidden
    >
      <path d="M6 6l12 12M6 18L18 6" />
    </svg>
  );
}
