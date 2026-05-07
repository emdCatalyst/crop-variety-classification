import { useEffect, useRef } from "react";
import { NavLink } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { unreadCount as fetchUnreadCount } from "@/api/notifications";
import { playNotificationChime } from "@/lib/chime";

const POLL_INTERVAL_MS = 60_000;
const STREAM_URL = "/api/v1/notifications/stream";

export default function NotificationsBell({
  unread,
  onUnreadChange,
}: {
  unread: number;
  onUnreadChange: (n: number) => void;
}) {
  const { t } = useTranslation();
  const lastUnreadRef = useRef(unread);

  useEffect(() => {
    lastUnreadRef.current = unread;
  }, [unread]);

  useEffect(() => {
    let cancelled = false;
    let es: EventSource | null = null;

    async function refresh(playOnIncrease: boolean) {
      try {
        const n = await fetchUnreadCount();
        if (cancelled) return;
        if (playOnIncrease && n > lastUnreadRef.current) {
          playNotificationChime();
        }
        onUnreadChange(n);
      } catch {
        // best-effort
      }
    }

    refresh(false);
    const pollId = window.setInterval(() => refresh(true), POLL_INTERVAL_MS);

    try {
      es = new EventSource(STREAM_URL, { withCredentials: true });
      es.addEventListener("notification", () => refresh(true));
      es.onerror = () => {
        // The browser auto-retries; let it.
      };
    } catch {
      // fall back to polling only
    }

    return () => {
      cancelled = true;
      window.clearInterval(pollId);
      es?.close();
    };
  }, [onUnreadChange]);

  return (
    <NavLink
      to="/notifications"
      className={({ isActive }) =>
        `relative inline-flex items-center justify-center w-9 h-9 rounded-md text-sm transition ${
          isActive ? "bg-brand-600 text-white" : "text-slate-700 hover:bg-slate-200"
        }`
      }
      aria-label={t("nav.notifications") ?? "Notifications"}
      title={t("nav.notifications") ?? ""}
    >
      <BellIcon />
      {unread > 0 && (
        <span
          className="absolute -top-1 -end-1 inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full bg-red-600 text-white text-[10px] font-semibold"
          aria-label={t("notifications.unread_aria", { count: unread }) ?? ""}
        >
          {unread > 99 ? "99+" : unread}
        </span>
      )}
    </NavLink>
  );
}

function BellIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9z" />
      <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
    </svg>
  );
}
