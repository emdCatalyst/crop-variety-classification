import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Archive, ArchiveRestore } from "lucide-react";
import {
  archiveConversation,
  attachmentUrl,
  listMessages,
  listThreads,
  markThreadRead,
  MessageRow,
  sendMessage,
  ThreadRow,
  unreadCount as fetchUnreadMessages,
} from "@/api/messages";
import { User } from "@/api/client";

const STREAM_URL = "/api/v1/messages/stream";
const MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024;

export default function MessagesPage({
  user,
  onUnreadChange,
}: {
  user: User;
  onUnreadChange?: (n: number) => void;
}) {
  const { t, i18n } = useTranslation();
  const isAdmin = user.role === "admin";
  const [threads, setThreads] = useState<ThreadRow[]>([]);
  const [activeCid, setActiveCid] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [body, setBody] = useState("");
  const [attachment, setAttachment] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const locale = i18n.language.split("-")[0];
  const listEndRef = useRef<HTMLDivElement | null>(null);

  const refreshThreads = useCallback(async () => {
    const t = await listThreads();
    setThreads(t);
    return t;
  }, []);

  useEffect(() => {
    refreshThreads().then((t) => {
      if (!isAdmin) {
        // The user has at most one *active* thread. The virtual fresh-thread
        // row carries an empty conversation_id which is fine — sending will
        // mint one server-side.
        const first = t.find((r) => !r.archived) ?? t[0];
        setActiveCid(first?.conversation_id ?? null);
      } else if (isAdmin && activeCid == null && t.length > 0) {
        const first = t.find((r) => !r.archived) ?? t[0];
        setActiveCid(first.conversation_id);
      }
    });
  }, [isAdmin, refreshThreads]);

  const activeThread = useMemo(
    () => threads.find((th) => th.conversation_id === activeCid) ?? null,
    [threads, activeCid]
  );

  const refreshMessages = useCallback(
    async (cid: string | null, withUserId: number | null) => {
      const data = await listMessages({
        conversationId: cid || undefined,
        // For non-admin we also pass withUserId so the server can route to
        // the support admin even when there's no live conversation yet.
        withUserId: withUserId ?? undefined,
      });
      setMessages(data);
    },
    []
  );

  // Use the primitive other_user_id everywhere effects depend on the active
  // thread — `threads` gets replaced on every refresh which would otherwise
  // create a new `activeThread` object reference and re-fire effects forever.
  const activeOtherId = activeThread?.other_user_id ?? null;

  useEffect(() => {
    if (!isAdmin) {
      refreshMessages(activeCid, null);
      return;
    }
    if (activeOtherId != null) {
      refreshMessages(activeCid, activeOtherId);
    } else {
      setMessages([]);
    }
  }, [activeCid, activeOtherId, isAdmin, refreshMessages]);

  useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length]);

  useEffect(() => {
    if (!activeCid && !isAdmin) return;
    if (isAdmin && activeOtherId == null) return;
    markThreadRead({
      conversationId: activeCid || undefined,
      withUserId: activeOtherId ?? undefined,
    })
      .then(async () => {
        // Refresh the threads list so the per-row unread badges update
        // immediately after viewing a thread. Safe now that the deps for
        // this effect are primitives (activeCid / activeOtherId), so a new
        // threads array doesn't bounce activeThread back into deps.
        await refreshThreads();
        try {
          const n = await fetchUnreadMessages();
          onUnreadChange?.(n);
        } catch {
          // best-effort
        }
      })
      .catch(() => undefined);
  }, [
    activeCid,
    activeOtherId,
    isAdmin,
    messages.length,
    onUnreadChange,
    refreshThreads,
  ]);

  // Stable mirror of activeCid + isAdmin so the SSE handler can read the
  // current values without us having to re-open the stream on every change.
  const activeCidRef = useRef(activeCid);
  const isAdminRef = useRef(isAdmin);
  useEffect(() => {
    activeCidRef.current = activeCid;
  }, [activeCid]);
  useEffect(() => {
    isAdminRef.current = isAdmin;
  }, [isAdmin]);

  useEffect(() => {
    let es: EventSource | null = null;
    try {
      es = new EventSource(STREAM_URL, { withCredentials: true });
      es.addEventListener("message", async () => {
        const t = await refreshThreads();
        if (!isAdminRef.current) {
          // Server filters archived threads for non-admins, so threads[0] is
          // either the active conversation or the virtual "fresh" placeholder
          // (cid=""). Always retarget to it — this is what makes an admin
          // archive immediately clear the user's chat history.
          const first = t[0];
          const newCid = first?.conversation_id ?? "";
          setActiveCid(newCid);
          await refreshMessages(newCid || null, null);
          return;
        }
        const cur = activeCidRef.current;
        if (cur) {
          const stillActive = t.find((r) => r.conversation_id === cur);
          if (stillActive) {
            await refreshMessages(cur, stillActive.other_user_id);
          }
        }
      });
      es.onerror = () => {
        // browser auto-retries
      };
    } catch {
      // ignore
    }
    return () => es?.close();
  }, [refreshMessages, refreshThreads]);

  const visibleThreads = useMemo(
    () => threads.filter((th) => (showArchived ? th.archived : !th.archived)),
    [threads, showArchived]
  );

  async function onArchiveActive() {
    if (!activeCid) return;
    try {
      await archiveConversation(activeCid, true);
      // Reset the right pane — admin shouldn't keep typing into a closed
      // conversation, and the archived list is opt-in via the toggle.
      setActiveCid(null);
      setMessages([]);
      await refreshThreads();
    } catch {
      // best-effort
    }
  }

  async function onSend(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const text = body.trim();
    if (!text && !attachment) return;
    if (attachment && attachment.size > MAX_ATTACHMENT_BYTES) {
      setError(t("messages.attachment_too_large"));
      return;
    }
    setSending(true);
    try {
      await sendMessage({
        body: text,
        recipientId: isAdmin ? activeThread?.other_user_id ?? null : null,
        attachment,
      });
      setBody("");
      setAttachment(null);
      const refreshed = await refreshThreads();
      // After sending, the active conversation may have changed (server may
      // have minted a new conversation_id if the previous one was archived).
      // Pick whatever live conversation the server now sees with this peer.
      const otherId = activeThread?.other_user_id ?? null;
      const next =
        refreshed.find(
          (r) =>
            !r.archived &&
            (otherId == null || r.other_user_id === otherId)
        ) ?? null;
      const nextCid = next?.conversation_id ?? null;
      setActiveCid(nextCid);
      await refreshMessages(nextCid, next?.other_user_id ?? null);
      onUnreadChange?.(0);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : t("messages.send_failed"));
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="flex flex-col gap-4 h-[calc(100vh-7.5rem)]">
      <header>
        <h1 className="text-2xl font-bold">{t("messages.title")}</h1>
        <p className="text-sm text-slate-600 mt-1">
          {isAdmin ? t("messages.subtitle_admin") : t("messages.subtitle_user")}
        </p>
      </header>

      <div
        className={`grid gap-4 flex-1 min-h-0 min-w-0 ${
          isAdmin ? "md:grid-cols-[280px_1fr]" : "md:grid-cols-1"
        }`}
      >
        {isAdmin && (
          <aside className="bg-white rounded-lg shadow-sm overflow-hidden flex flex-col min-h-0 min-w-0">
            <div className="p-3 border-b border-slate-200 flex items-center justify-between gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                {showArchived ? t("messages.archived_threads") : t("messages.threads")}
              </span>
              <button
                type="button"
                onClick={() => setShowArchived((v) => !v)}
                className="inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded-md hover:bg-slate-100 text-slate-600"
                title={t("messages.toggle_archived") ?? ""}
              >
                {showArchived ? <ArchiveRestore size={12} /> : <Archive size={12} />}
                {showArchived ? t("messages.show_active") : t("messages.show_archived")}
              </button>
            </div>
            {visibleThreads.length === 0 ? (
              <p className="p-4 text-sm text-slate-500">
                {showArchived
                  ? t("messages.no_archived_threads")
                  : t("messages.no_threads")}
              </p>
            ) : (
              <ul className="divide-y divide-slate-200 flex-1 overflow-y-auto min-h-0">
                {visibleThreads.map((th) => {
                  const isActive = th.conversation_id === activeCid;
                  return (
                    <li key={th.conversation_id}>
                      <button
                        type="button"
                        onClick={() => setActiveCid(th.conversation_id)}
                        className={`w-full text-start p-3 hover:bg-slate-50 ${
                          isActive ? "bg-brand-50" : ""
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-medium truncate">{th.other_user_name}</span>
                          {th.unread_count > 0 && (
                            <span className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full bg-red-600 text-white text-[10px] font-semibold">
                              {th.unread_count > 99 ? "99+" : th.unread_count}
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-slate-500 truncate mt-0.5">
                          {th.last_body ??
                            (th.last_has_attachment ? t("messages.attachment_label") : "—")}
                        </div>
                        <div className="text-[10px] text-slate-400 mt-1">
                          {new Date(th.last_at).toLocaleString(locale)}
                        </div>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </aside>
        )}

        <section className="bg-white rounded-lg shadow-sm flex flex-col h-full min-h-0 min-w-0 overflow-hidden">
          <div className="p-3 border-b border-slate-200 flex items-center justify-between gap-2">
            <div className="font-medium min-w-0 truncate">
              {isAdmin
                ? activeThread?.other_user_name ?? t("messages.select_thread")
                : t("messages.support_team")}
            </div>
            {isAdmin && activeThread && !activeThread.archived ? (
              <button
                type="button"
                onClick={() => onArchiveActive()}
                className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md border border-slate-300 hover:bg-slate-50 shrink-0"
              >
                <Archive size={12} aria-hidden />
                {t("messages.archive")}
              </button>
            ) : !isAdmin ? (
              <span className="text-xs text-slate-500 shrink-0">
                {t("messages.support_hint")}
              </span>
            ) : null}
          </div>

          <div
            dir="ltr"
            className="flex-1 overflow-y-auto min-h-0 p-4 space-y-3 bg-slate-50"
          >
            {messages.length === 0 ? (
              <p className="text-sm text-slate-500 text-center mt-10">
                {isAdmin
                  ? activeThread
                    ? t("messages.thread_empty")
                    : t("messages.select_thread")
                  : t("messages.start_conversation")}
              </p>
            ) : (
              messages.map((m) => {
                const mine = m.sender_id === user.id;
                return (
                  <div
                    key={m.id}
                    className={`flex w-full min-w-0 ${
                      mine ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`min-w-0 max-w-[80%] rounded-lg px-3 py-2 shadow-sm text-sm overflow-hidden ${
                        mine
                          ? "bg-brand-600 text-white"
                          : "bg-white text-slate-800 border border-slate-200"
                      }`}
                    >
                      {!mine && (
                        <div className="text-[10px] font-semibold opacity-70 mb-0.5">
                          {m.sender_name}
                        </div>
                      )}
                      {m.body && (
                        <div
                          dir="auto"
                          className="whitespace-pre-wrap break-words [overflow-wrap:anywhere]"
                        >
                          {m.body}
                        </div>
                      )}
                      {m.has_attachment && (
                        <a
                          href={attachmentUrl(m.id)}
                          target="_blank"
                          rel="noreferrer"
                          className={`block mt-2 ${mine ? "" : ""}`}
                        >
                          <img
                            src={attachmentUrl(m.id)}
                            alt={m.attachment_name ?? "attachment"}
                            className="max-h-48 rounded-md border border-slate-200 bg-white"
                          />
                        </a>
                      )}
                      <div
                        className={`text-[10px] mt-1 ${
                          mine ? "text-white/70" : "text-slate-400"
                        }`}
                      >
                        {new Date(m.created_at).toLocaleString(locale)}
                      </div>
                    </div>
                  </div>
                );
              })
            )}
            <div ref={listEndRef} />
          </div>

          {isAdmin && activeThread?.archived ? (
            <div className="border-t border-slate-200 p-3 bg-slate-50">
              <p className="text-xs text-slate-600">
                {t("messages.archived_banner")}
              </p>
            </div>
          ) : (activeThread || !isAdmin) && (
            <form
              onSubmit={onSend}
              className="border-t border-slate-200 p-3 space-y-2 bg-white"
            >
              {attachment && (
                <div className="flex items-center justify-between text-xs bg-slate-100 rounded px-2 py-1">
                  <span className="truncate">{attachment.name}</span>
                  <button
                    type="button"
                    onClick={() => setAttachment(null)}
                    className="ms-2 text-red-600 hover:underline"
                  >
                    {t("common.delete")}
                  </button>
                </div>
              )}
              {error && <div className="text-xs text-red-600">{error}</div>}
              <div className="flex items-end gap-2 min-w-0">
                <textarea
                  value={body}
                  onChange={(e) => setBody(e.target.value)}
                  placeholder={t("messages.input_placeholder") ?? ""}
                  rows={2}
                  className="flex-1 min-w-0 rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500 resize-y"
                />
                <label className="shrink-0 cursor-pointer rounded-md border border-slate-300 px-3 py-2 text-xs hover:bg-slate-50">
                  📎
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    className="hidden"
                    onChange={(e) => setAttachment(e.target.files?.[0] ?? null)}
                  />
                </label>
                <button
                  type="submit"
                  disabled={sending || (!body.trim() && !attachment)}
                  className="shrink-0 whitespace-nowrap rounded-md bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
                >
                  {sending ? t("common.loading") : t("messages.send")}
                </button>
              </div>
            </form>
          )}
        </section>
      </div>
    </div>
  );
}
