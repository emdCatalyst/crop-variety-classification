import { FormEvent, useEffect, useRef, useState } from "react";
import * as RD from "@radix-ui/react-dialog";
import { useTranslation } from "react-i18next";
import { Check, Send, X } from "lucide-react";
import { notifyUser } from "@/api/admin";

export default function NotifyUserDialog({
  open,
  onOpenChange,
  userId,
  userName,
  userEmail,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  userId: number | null;
  userName: string;
  userEmail: string;
}) {
  const { t } = useTranslation();
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const closeTimer = useRef<number | null>(null);

  useEffect(() => {
    if (open) {
      setTitle("");
      setBody("");
      setError(null);
      setSent(false);
    } else if (closeTimer.current !== null) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  }, [open]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (userId == null) return;
    setPending(true);
    setError(null);
    try {
      await notifyUser({ user_id: userId, title: title.trim(), body: body.trim() });
      setSent(true);
      closeTimer.current = window.setTimeout(() => onOpenChange(false), 1200);
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response
        ?.data?.detail;
      setError(typeof detail === "string" ? detail : t("admin.notify.failed"));
    } finally {
      setPending(false);
    }
  }

  return (
    <RD.Root open={open} onOpenChange={onOpenChange}>
      <RD.Portal>
        <RD.Overlay className="fixed inset-0 z-50 bg-slate-900/50 animate-fade-in" />
        <RD.Content className="fixed inset-0 z-50 flex items-center justify-center p-4 focus:outline-none">
          <div className="w-[min(92vw,32rem)] max-h-[calc(100vh-2rem)] overflow-y-auto rounded-xl bg-white shadow-xl border border-slate-200 p-5 animate-slide-up">
          <div className="flex items-start justify-between gap-3 mb-3">
            <div className="min-w-0">
              <RD.Title className="text-lg font-semibold text-slate-900">
                {t("admin.notify.title")}
              </RD.Title>
              <RD.Description className="text-xs text-slate-500 mt-0.5 truncate">
                {userName} · {userEmail}
              </RD.Description>
            </div>
            <RD.Close
              className="shrink-0 rounded-md p-1 text-slate-500 hover:bg-slate-100"
              aria-label={t("common.cancel") ?? "Close"}
            >
              <X size={16} aria-hidden />
            </RD.Close>
          </div>

          {sent ? (
            <div className="py-6 text-center animate-fade-in">
              <div className="mx-auto w-12 h-12 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center">
                <Check size={22} aria-hidden />
              </div>
              <p className="text-sm font-semibold text-slate-900 mt-3">
                {t("admin.notify.sent")}
              </p>
              <p className="text-xs text-slate-500 mt-1">
                {t("admin.notify.sent_hint", { name: userName })}
              </p>
            </div>
          ) : (
            <form onSubmit={onSubmit} className="space-y-3">
              <label className="block text-sm">
                {t("admin.broadcast.title_label")}
                <input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  required
                  maxLength={200}
                  autoFocus
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
              {error && (
                <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-md p-2">
                  {error}
                </div>
              )}
              <div className="flex justify-end gap-2">
                <RD.Close asChild>
                  <button
                    type="button"
                    className="text-sm rounded-md px-3 py-2 hover:bg-slate-100"
                  >
                    {t("common.cancel")}
                  </button>
                </RD.Close>
                <button
                  type="submit"
                  disabled={pending || !title.trim() || !body.trim()}
                  className="inline-flex items-center gap-1.5 rounded-md bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 text-sm font-semibold disabled:opacity-50"
                >
                  <Send size={14} aria-hidden />
                  {pending ? t("common.loading") : t("admin.notify.send")}
                </button>
              </div>
            </form>
          )}
          </div>
        </RD.Content>
      </RD.Portal>
    </RD.Root>
  );
}
