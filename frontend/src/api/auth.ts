import { api, User } from "./client";

export async function fetchMe(): Promise<User | null> {
  try {
    const { data } = await api.get<User>("/auth/me");
    return data;
  } catch {
    return null;
  }
}

export async function login(email: string, password: string): Promise<User> {
  const { data } = await api.post<User>("/auth/login", { email, password });
  return data;
}

export async function signup(payload: {
  email: string;
  password: string;
  display_name: string;
  language?: string;
}): Promise<User> {
  const { data } = await api.post<User>("/auth/signup", { language: "en", ...payload });
  return data;
}

export async function logout(): Promise<void> {
  await api.post("/auth/logout");
}
