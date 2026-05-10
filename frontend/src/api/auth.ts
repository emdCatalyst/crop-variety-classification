import { api, User } from "./client";

export type SignupResponse = {
  email: string;
  status: "verification_required";
};

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
}): Promise<SignupResponse> {
  const { data } = await api.post<SignupResponse>("/auth/signup", { language: "en", ...payload });
  return data;
}

export async function logout(): Promise<void> {
  await api.post("/auth/logout");
}

export async function verifyEmail(email: string, code: string): Promise<User> {
  const { data } = await api.post<User>("/auth/verify-email", { email, code });
  return data;
}

export async function resendVerification(email: string): Promise<void> {
  await api.post("/auth/resend-verification", { email });
}

export async function forgotPassword(email: string): Promise<void> {
  await api.post("/auth/forgot-password", { email });
}

export async function resetPassword(
  email: string,
  code: string,
  newPassword: string,
): Promise<User> {
  const { data } = await api.post<User>("/auth/reset-password", {
    email,
    code,
    new_password: newPassword,
  });
  return data;
}
