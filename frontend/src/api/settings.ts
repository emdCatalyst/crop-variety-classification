import { api, User } from "./client";

export type ProfileUpdate = {
  display_name?: string;
  language?: string;
};

export async function updateProfile(payload: ProfileUpdate): Promise<User> {
  const { data } = await api.patch<User>("/settings/profile", payload);
  return data;
}

export async function changePassword(payload: {
  current_password: string;
  new_password: string;
}): Promise<void> {
  await api.post("/settings/password", payload);
}
