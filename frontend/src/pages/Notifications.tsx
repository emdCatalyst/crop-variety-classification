import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  deleteNotification,
  listNotifications,
  markAllRead,
  markRead,
  Notification,
} from "@/api/notifications";
import { useNotificationsStream } from "@/hooks/useNotificationsStream";

export default function NotificationsPage({
  onUnreadChange,
}: {
  onUnreadChange?: (n: number) => void;
}) {
  const { t, i18n } = useTranslation();
  const [items, setItems] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);
  const locale = i18n.language.split("-")[0];

  const load = useCallback(
    async (showSpinner: boolean) => {
      if (showSpinner) setLoading(true);
      try {
        const data = await listNotifications();
        setItems(data);
        onUnreadChange?.(data.filter((n) => n.read_at === null).length);
      } finally {
        if (showSpinner) setLoading(false);
      }
    },
    [onUnreadChange]
  );

  const refresh = useCallback(() => {
    void load(true);
  }, [load]);

  const refreshSilent = useCallback(() => {
    void load(false);
  }, [load]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Live-update when a new notification lands while the page is open.
  useNotificationsStream(refreshSilent);

  async function onMarkRead(id: number) {
    await markRead(id);
    setItems((items) =>
      items.map((n) => (n.id === id ? { ...n, read_at: new Date().toISOString() } : n))
    );
    onUnreadChange?.(items.filter((n) => n.read_at === null && n.id !== id).length);
  }

  async function onDelete(id: number) {
    await deleteNotification(id);
    setItems((items) => items.filter((n) => n.id !== id));
    onUnreadChange?.(items.filter((n) => n.read_at === null && n.id !== id).length);
  }

  async function onMarkAllRead() {
    await markAllRead();
    const stamp = new Date().toISOString();
    setItems((items) => items.map((n) => (n.read_at ? n : { ...n, read_at: stamp })));
    onUnreadChange?.(0);
  }

  const unread = items.filter((n) => n.read_at === null).length;

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">{t("notifications.title")}</h1>
          <p className="text-sm text-slate-600 mt-1">{t("notifications.subtitle")}</p>
        </div>
        {unread > 0 && (
          <button
            onClick={onMarkAllRead}
            className="text-sm rounded-md bg-slate-100 hover:bg-slate-200 px-3 py-1.5"
          >
            {t("notifications.mark_all_read")}
          </button>
        )}
      </header>

      {loading ? (
        <p className="text-sm text-slate-500">{t("common.loading")}</p>
      ) : items.length === 0 ? (
        <p className="text-sm text-slate-500">{t("notifications.empty")}</p>
      ) : (
        <ul className="bg-white rounded-lg shadow-sm divide-y divide-slate-200">
          {items.map((n) => (
            <li key={n.id} className={`p-4 flex gap-3 ${n.read_at ? "" : "bg-emerald-50/40"}`}>
              <div className="mt-1.5 shrink-0">
                <span
                  className={`inline-block w-2.5 h-2.5 rounded-full ${
                    n.read_at ? "bg-slate-300" : "bg-emerald-500"
                  }`}
                  aria-hidden
                />
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-medium">
                  {n.i18n_key
                    ? t(`notifications.system.${n.i18n_key}.title`, {
                        ...(n.i18n_params ?? {}),
                        defaultValue: n.title,
                      })
                    : n.title}
                </div>
                <div className="text-sm text-slate-700 mt-0.5">
                  {n.i18n_key
                    ? t(`notifications.system.${n.i18n_key}.body`, {
                        ...(n.i18n_params ?? {}),
                        defaultValue: n.body,
                      })
                    : n.body}
                </div>
                <div className="text-xs text-slate-500 mt-1 flex items-center gap-3">
                  <span>{new Date(n.created_at).toLocaleString(locale)}</span>
                  {n.analysis_id && (
                    <Link
                      to={`/analyses/${n.analysis_id}`}
                      className="text-brand-700 hover:underline"
                    >
                      {t("notifications.view_analysis")}
                    </Link>
                  )}
                </div>
              </div>
              <div className="flex items-start gap-2">
                {!n.read_at && (
                  <button
                    onClick={() => onMarkRead(n.id)}
                    className="text-xs px-2 py-1 rounded-md border border-slate-300 hover:bg-slate-50"
                  >
                    {t("notifications.mark_read")}
                  </button>
                )}
                <button
                  onClick={() => onDelete(n.id)}
                  className="text-xs px-2 py-1 rounded-md text-red-600 hover:bg-red-50"
                >
                  {t("common.delete")}
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
