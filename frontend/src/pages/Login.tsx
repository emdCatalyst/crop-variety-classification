import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { login } from "@/api/auth";
import { User } from "@/api/client";
import LanguageSwitcher from "@/components/LanguageSwitcher";

export default function LoginPage({ onSignIn }: { onSignIn: (u: User) => void }) {
  const { t } = useTranslation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const navigate = useNavigate();

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setPending(true);
    setError(null);
    try {
      const u = await login(email, password);
      onSignIn(u);
      navigate("/dashboard");
    } catch (err: any) {
      setError(err.response?.data?.detail ?? t("auth.login_failed"));
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-brand-50 to-white">
      <form onSubmit={onSubmit} className="w-full max-w-sm bg-white rounded-xl shadow p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-brand-700">{t("auth.sign_in")}</h1>
          <LanguageSwitcher />
        </div>
        <p className="text-sm text-slate-600">{t("auth.sign_in_subtitle")}</p>

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
          {t("auth.password")}
          <input
            type="password"
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
          {pending ? t("auth.signing_in") : t("auth.sign_in")}
        </button>

        <div className="text-sm text-slate-600 text-center">
          {t("auth.new_here")}{" "}
          <Link to="/signup" className="text-brand-700 hover:underline">
            {t("auth.create_account_link")}
          </Link>
        </div>
      </form>
    </div>
  );
}
