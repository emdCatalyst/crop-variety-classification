import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { signup } from "@/api/auth";
import LanguageSwitcher from "@/components/LanguageSwitcher";
import { apiErrorMessage } from "@/lib/apiError";

export default function SignupPage() {
  const { t, i18n } = useTranslation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const navigate = useNavigate();

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setPending(true);
    setError(null);
    try {
      await signup({
        email,
        password,
        display_name: name,
        language: i18n.language.split("-")[0],
      });
      navigate(`/verify-email?email=${encodeURIComponent(email)}`);
    } catch (err) {
      setError(apiErrorMessage(err, t("auth.signup_failed")));
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-brand-50 to-white">
      <form onSubmit={onSubmit} className="w-full max-w-sm bg-white rounded-xl shadow p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-brand-700">{t("auth.create_account")}</h1>
          <LanguageSwitcher />
        </div>

        <label className="block text-sm">
          {t("auth.display_name")}
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500"
          />
        </label>

        <label className="block text-sm">
          {t("auth.email")}
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500"
          />
        </label>

        <label className="block text-sm">
          {t("auth.password_min")}
          <input
            type="password"
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500"
          />
        </label>

        {error && <div className="text-sm text-red-600">{error}</div>}

        <button
          type="submit"
          disabled={pending}
          className="w-full rounded-md bg-brand-600 hover:bg-brand-700 text-white py-2 text-sm font-medium disabled:opacity-50"
        >
          {pending ? t("auth.creating") : t("auth.create_account")}
        </button>

        <div className="text-sm text-slate-600 text-center">
          {t("auth.already_member")}{" "}
          <Link to="/login" className="text-brand-700 hover:underline">
            {t("auth.sign_in")}
          </Link>
        </div>
      </form>
    </div>
  );
}
