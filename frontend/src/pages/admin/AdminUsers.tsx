import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { AdminUser, deleteUser, listUsers, updateUser } from "@/api/admin";
import { User } from "@/api/client";

export default function AdminUsersPage({ currentUser }: { currentUser: User }) {
  const { t, i18n } = useTranslation();
  const [rows, setRows] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const locale = i18n.language.split("-")[0];

  async function refresh() {
    setLoading(true);
    try {
      setRows(await listUsers());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function patch(
    id: number,
    p: Partial<Pick<AdminUser, "display_name" | "role" | "is_active">>
  ) {
    setError(null);
    try {
      const updated = await updateUser(id, p);
      setRows((rs) => rs.map((r) => (r.id === id ? updated : r)));
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response
        ?.data?.detail;
      setError(typeof detail === "string" ? detail : t("admin.users.update_failed"));
    }
  }

  async function onDelete(u: AdminUser) {
    if (!confirm(t("admin.users.confirm_delete", { email: u.email }) ?? "")) return;
    try {
      await deleteUser(u.id);
      setRows((rs) => rs.filter((r) => r.id !== u.id));
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response
        ?.data?.detail;
      setError(typeof detail === "string" ? detail : t("admin.users.delete_failed"));
    }
  }

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-bold">{t("admin.nav.users")}</h1>
      </header>
      {error && (
        <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-md p-2">
          {error}
        </div>
      )}

      {loading ? (
        <p className="text-sm text-slate-500">{t("common.loading")}</p>
      ) : (
        <ul className="bg-white rounded-lg shadow-sm divide-y divide-slate-200">
          {rows.map((u) => {
            const isSelf = u.id === currentUser.id;
            return (
              <li key={u.id} className="p-3 flex flex-col md:flex-row md:items-center gap-3">
                <div className="min-w-0 md:flex-1">
                  <div className="font-medium truncate">{u.display_name}</div>
                  <div className="text-xs text-slate-500 truncate">{u.email}</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    {new Date(u.created_at).toLocaleString(locale)}
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <select
                    value={u.role}
                    disabled={isSelf}
                    onChange={(e) => patch(u.id, { role: e.target.value as "user" | "admin" })}
                    className="text-xs rounded-md border border-slate-300 px-2 py-1 bg-white disabled:opacity-50"
                  >
                    <option value="user">{t("admin.users.role_user")}</option>
                    <option value="admin">{t("admin.users.role_admin")}</option>
                  </select>
                  <label className="inline-flex items-center gap-1.5 text-xs text-slate-600">
                    <input
                      type="checkbox"
                      checked={u.is_active}
                      disabled={isSelf}
                      onChange={(e) => patch(u.id, { is_active: e.target.checked })}
                    />
                    {t("admin.users.active")}
                  </label>
                  <button
                    onClick={() => onDelete(u)}
                    disabled={isSelf}
                    className="text-xs px-2 py-1 rounded-md text-red-600 hover:bg-red-50 disabled:opacity-50"
                  >
                    {t("common.delete")}
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
