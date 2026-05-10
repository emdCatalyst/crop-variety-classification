import { FormEvent, useState } from "react";
import { useTranslation } from "react-i18next";
import { broadcast } from "@/api/admin";

export default function AdminBroadcastPage() {
  const { t } = useTranslation();
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [onlyActive, setOnlyActive] = useState(true);
  const [pending, setPending] = useState(false);
  const [feedback, setFeedback] = useState<{ kind: "ok" | "err"; msg: string } | null>(
    null
  );

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setPending(true);
    setFeedback(null);
    try {
      const res = await broadcast({
        title: title.trim(),
        body: body.trim(),
        only_active: onlyActive,
      });
      setFeedback({
        kind: "ok",
        msg: t("admin.broadcast.sent", { count: res.sent }),
      });
      setTitle("");
      setBody("");
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response
        ?.data?.detail;
      setFeedback({
        kind: "err",
        msg: typeof detail === "string" ? detail : t("admin.broadcast.failed"),
      });
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-bold">{t("admin.nav.broadcast")}</h1>
      </header>
      <form
        onSubmit={onSubmit}
        className="bg-white rounded-lg shadow-sm p-5 space-y-4"
      >
        <p className="text-sm text-slate-600">{t("admin.broadcast.hint")}</p>

      <label className="block text-sm">
        {t("admin.broadcast.title_label")}
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          required
          maxLength={200}
          className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500"
        />
      </label>

      <label className="block text-sm">
        {t("admin.broadcast.body_label")}
        <textarea
          value={body}
          onChange={(e) => setBody(e.target.value)}
          required
          rows={5}
          className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500 resize-y"
        />
      </label>

      <label className="inline-flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={onlyActive}
          onChange={(e) => setOnlyActive(e.target.checked)}
        />
        {t("admin.broadcast.only_active")}
      </label>

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={pending || !title.trim() || !body.trim()}
          className="rounded-md bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
        >
          {pending ? t("common.loading") : t("admin.broadcast.send")}
        </button>
        {feedback && (
          <span
            className={`text-sm ${
              feedback.kind === "ok" ? "text-emerald-700" : "text-red-600"
            }`}
          >
            {feedback.msg}
          </span>
        )}
        </div>
      </form>
    </div>
  );
}
