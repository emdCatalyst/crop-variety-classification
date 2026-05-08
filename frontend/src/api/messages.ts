import { api } from "./client";

export type MessageRow = {
  id: number;
  sender_id: number;
  sender_name: string;
  recipient_id: number;
  body: string | null;
  has_attachment: boolean;
  attachment_name: string | null;
  attachment_mime: string | null;
  read_at: string | null;
  created_at: string;
  archived: boolean;
};

export type ThreadRow = {
  thread_key: string;
  other_user_id: number;
  other_user_name: string;
  other_user_role: string;
  last_body: string | null;
  last_has_attachment: boolean;
  last_at: string;
  unread_count: number;
  archived: boolean;
};

export async function listThreads(): Promise<ThreadRow[]> {
  const { data } = await api.get<ThreadRow[]>("/messages/threads");
  return data;
}

export async function listMessages(withUserId?: number | null): Promise<MessageRow[]> {
  const { data } = await api.get<MessageRow[]>("/messages", {
    params: withUserId != null ? { with_user_id: withUserId } : undefined,
  });
  return data;
}

export async function sendMessage(payload: {
  body: string;
  recipientId?: number | null;
  attachment?: File | null;
}): Promise<MessageRow> {
  const fd = new FormData();
  if (payload.body) fd.append("body", payload.body);
  if (payload.recipientId != null) fd.append("recipient_id", String(payload.recipientId));
  if (payload.attachment) fd.append("attachment", payload.attachment, payload.attachment.name);
  const { data } = await api.post<MessageRow>("/messages", fd);
  return data;
}

export async function unreadCount(): Promise<number> {
  const { data } = await api.get<{ unread: number }>("/messages/unread-count");
  return data.unread;
}

export async function markThreadRead(withUserId?: number | null): Promise<void> {
  await api.post(
    "/messages/read",
    null,
    withUserId != null ? { params: { with_user_id: withUserId } } : undefined
  );
}

export function attachmentUrl(messageId: number): string {
  return `/api/v1/messages/${messageId}/attachment`;
}
