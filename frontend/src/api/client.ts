import axios from "axios";

export const api = axios.create({
  baseURL: "/api/v1",
  withCredentials: true,
});

export type User = {
  id: number;
  email: string;
  display_name: string;
  role: string;
  language: string;
};

export type Analysis = {
  id: number;
  source_name: string;
  status: "queued" | "processing" | "completed" | "failed";
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type AnalysisResult = {
  predicted_crop: string;
  health_status: string;
  trajectory: string;
  advice: string;
  mean_confidence: number;
  mean_ndvi: number;
  mean_ndre: number;
  mean_savi: number;
  temporal_trend: number;
  map_url: string;
  class_distribution: Record<string, number>;
  farmer_notes: string | null;
  observed_at: string | null;
};

export type AnalysisDetail = Analysis & { result: AnalysisResult | null };
