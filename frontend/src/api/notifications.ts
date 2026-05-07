import { api } from "./client";

export type Notification = {
  id: number;
  kind: string;
  title: string;
  body: string;
  analysis_id: number | null;
  read_at: string | null;
  created_at: string;
};

export async function listNotifications(): Promise<Notification[]> {
  const { data } = await api.get<Notification[]>("/notifications");
  return data;
}

export async function unreadCount(): Promise<number> {
  const { data } = await api.get<{ unread: number }>("/notifications/unread-count");
  return data.unread;
}

export async function markRead(id: number): Promise<void> {
  await api.post(`/notifications/${id}/read`);
}

export async function markAllRead(): Promise<void> {
  await api.post("/notifications/read-all");
}

export async function deleteNotification(id: number): Promise<void> {
  await api.delete(`/notifications/${id}`);
}
