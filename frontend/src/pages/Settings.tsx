import { FormEvent, useState } from "react";
import { useTranslation } from "react-i18next";
import { User } from "@/api/client";
import { changePassword, updateProfile } from "@/api/settings";
import { SUPPORTED_LANGUAGES, SupportedLanguage } from "@/i18n/i18n";
import { Select } from "@/components/Select";

export default function SettingsPage({
  user,
  onUserChange,
}: {
  user: User;
  onUserChange: (u: User) => void;
}) {
  const { t, i18n } = useTranslation();
  const [displayName, setDisplayName] = useState(user.display_name);
  const [language, setLanguage] = useState<SupportedLanguage>(
    (user.language as SupportedLanguage) ?? "en"
  );
  const [profileMsg, setProfileMsg] = useState<{ kind: "ok" | "err"; msg: string } | null>(null);
  const [profileSaving, setProfileSaving] = useState(false);

  const [currentPwd, setCurrentPwd] = useState("");
  const [newPwd, setNewPwd] = useState("");
  const [pwdMsg, setPwdMsg] = useState<{ kind: "ok" | "err"; msg: string } | null>(null);
  const [pwdSaving, setPwdSaving] = useState(false);

  async function saveProfile(e: FormEvent) {
    e.preventDefault();
    setProfileSaving(true);
    setProfileMsg(null);
    try {
      const updated = await updateProfile({ display_name: displayName, language });
      onUserChange(updated);
      if (i18n.language.split("-")[0] !== language) await i18n.changeLanguage(language);
      setProfileMsg({ kind: "ok", msg: t("settings.profile_saved") });
    } catch (err: any) {
      setProfileMsg({
        kind: "err",
        msg: err?.response?.data?.detail ?? t("common.error"),
      });
    } finally {
      setProfileSaving(false);
    }
  }

  async function savePassword(e: FormEvent) {
    e.preventDefault();
    setPwdSaving(true);
    setPwdMsg(null);
    try {
      await changePassword({ current_password: currentPwd, new_password: newPwd });
      setCurrentPwd("");
      setNewPwd("");
      setPwdMsg({ kind: "ok", msg: t("settings.password_changed") });
    } catch (err: any) {
      setPwdMsg({
        kind: "err",
        msg: err?.response?.data?.detail ?? t("settings.password_failed"),
      });
    } finally {
      setPwdSaving(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">{t("settings.title")}</h1>

      <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
        <h2 className="font-semibold mb-4">{t("settings.profile")}</h2>
        <form onSubmit={saveProfile} className="space-y-4">
          <label className="block text-sm">
            {t("settings.display_name")}
            <input
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              required
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500"
            />
          </label>
          <div className="text-sm">
            <label className="block mb-1">{t("settings.language")}</label>
            <Select<SupportedLanguage>
              value={language}
              onValueChange={setLanguage}
              triggerClassName="w-full justify-between"
              options={SUPPORTED_LANGUAGES.map((lng) => ({
                value: lng,
                label: t(`settings.language_${lng}`),
              }))}
            />
          </div>
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={profileSaving}
              className="rounded-md bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
            >
              {profileSaving ? t("common.loading") : t("common.save_changes")}
            </button>
            {profileMsg && (
              <span
                className={`text-sm ${
                  profileMsg.kind === "ok" ? "text-emerald-700" : "text-red-600"
                }`}
              >
                {profileMsg.msg}
              </span>
            )}
          </div>
        </form>
      </section>

      <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
        <h2 className="font-semibold mb-4">{t("settings.change_password")}</h2>
        <form onSubmit={savePassword} className="space-y-4">
          <label className="block text-sm">
            {t("settings.current_password")}
            <input
              type="password"
              value={currentPwd}
              onChange={(e) => setCurrentPwd(e.target.value)}
              required
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500"
            />
          </label>
          <label className="block text-sm">
            {t("settings.new_password")}
            <input
              type="password"
              minLength={8}
              value={newPwd}
              onChange={(e) => setNewPwd(e.target.value)}
              required
              className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500"
            />
          </label>
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={pwdSaving}
              className="rounded-md bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
            >
              {pwdSaving ? t("common.loading") : t("common.save_changes")}
            </button>
            {pwdMsg && (
              <span
                className={`text-sm ${
                  pwdMsg.kind === "ok" ? "text-emerald-700" : "text-red-600"
                }`}
              >
                {pwdMsg.msg}
              </span>
            )}
          </div>
        </form>
      </section>
    </div>
  );
}
