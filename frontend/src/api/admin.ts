import { api } from "./client";

export type AdminStats = {
  users_total: number;
  users_active: number;
  admins: number;
  analyses_total: number;
  analyses_by_status: {
    queued: number;
    processing: number;
    completed: number;
    failed: number;
  };
  notifications_total: number;
  messages_total: number;
};

export type AdminActivityRow = {
  kind: "analysis" | "user" | "message";
  title: string;
  detail: string | null;
  at: string;
};

export type AdminUser = {
  id: number;
  email: string;
  display_name: string;
  role: "user" | "admin";
  is_active: boolean;
  language: string;
  created_at: string;
};

export type AdminAnalysisRow = {
  id: number;
  source_name: string;
  status: "queued" | "processing" | "completed" | "failed";
  error: string | null;
  created_at: string;
  updated_at: string;
  user_id: number;
  user_email: string;
  user_display_name: string;
  has_result: boolean;
};

export async function getStats(): Promise<AdminStats> {
  const { data } = await api.get<AdminStats>("/admin/stats");
  return data;
}

export async function getActivity(limit = 15): Promise<AdminActivityRow[]> {
  const { data } = await api.get<AdminActivityRow[]>("/admin/stats/activity", {
    params: { limit },
  });
  return data;
}

export async function listUsers(): Promise<AdminUser[]> {
  const { data } = await api.get<AdminUser[]>("/admin/users");
  return data;
}

export async function updateUser(
  id: number,
  patch: Partial<Pick<AdminUser, "display_name" | "role" | "is_active">>
): Promise<AdminUser> {
  const { data } = await api.patch<AdminUser>(`/admin/users/${id}`, patch);
  return data;
}

export async function deleteUser(id: number): Promise<void> {
  await api.delete(`/admin/users/${id}`);
}

export async function listAnalyses(params?: { status?: string }): Promise<AdminAnalysisRow[]> {
  const { data } = await api.get<AdminAnalysisRow[]>("/admin/analyses", { params });
  return data;
}

export async function deleteAnalysis(id: number): Promise<void> {
  await api.delete(`/admin/analyses/${id}`);
}

export type AdminTimeseriesPoint = {
  date: string;
  analyses: number;
  new_users: number;
};

export async function getAdminTimeseries(days = 30): Promise<AdminTimeseriesPoint[]> {
  const { data } = await api.get<AdminTimeseriesPoint[]>("/admin/stats/timeseries", {
    params: { days },
  });
  return data;
}

export async function broadcast(payload: {
  title: string;
  body: string;
  only_active?: boolean;
}): Promise<{ sent: number }> {
  const { data } = await api.post<{ sent: number }>("/admin/notifications/broadcast", payload);
  return data;
}

export async function notifyUser(payload: {
  user_id: number;
  title: string;
  body: string;
}): Promise<{ id: number }> {
  const { data } = await api.post<{ id: number }>("/admin/notifications/notify", payload);
  return data;
}
