import { api } from "./client";

export type ReportRow = {
  id: number;
  source_name: string;
  status: "queued" | "processing" | "completed" | "failed";
  created_at: string;
  predicted_crop: string | null;
  health_status: string | null;
  farmer_notes: string | null;
  observed_at: string | null;
  has_result: boolean;
};

export async function listReports(params?: { status?: string }): Promise<ReportRow[]> {
  const { data } = await api.get<ReportRow[]>("/reports", { params });
  return data;
}

export async function updateNotes(
  id: number,
  payload: { farmer_notes: string | null; observed_at?: string | null }
): Promise<void> {
  await api.patch(`/reports/${id}/notes`, payload);
}

export function pdfUrl(id: number, lang: string): string {
  return `/api/v1/reports/${id}/pdf?lang=${encodeURIComponent(lang)}`;
}
