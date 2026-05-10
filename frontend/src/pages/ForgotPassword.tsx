import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { forgotPassword, resetPassword } from "@/api/auth";
import { User } from "@/api/client";
import LanguageSwitcher from "@/components/LanguageSwitcher";
import { apiErrorMessage } from "@/lib/apiError";

export default function ForgotPasswordPage({ onSignIn }: { onSignIn: (u: User) => void }) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [step, setStep] = useState<"request" | "reset">("request");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function onRequest(e: FormEvent) {
    e.preventDefault();
    setPending(true);
    setError(null);
    setInfo(null);
    try {
      await forgotPassword(email);
      setInfo(t("auth.forgot_request_sent"));
      setStep("reset");
    } catch (err) {
      setError(apiErrorMessage(err, t("auth.signup_failed")));
    } finally {
      setPending(false);
    }
  }

  async function onReset(e: FormEvent) {
    e.preventDefault();
    setPending(true);
    setError(null);
    try {
      const u = await resetPassword(email, code, newPassword);
      onSignIn(u);
      navigate("/dashboard");
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      if (detail === "invalid_or_expired_code") {
        setError(t("auth.code_invalid"));
      } else {
        setError(apiErrorMessage(err, t("auth.code_invalid")));
      }
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-brand-50 to-white">
      <form
        onSubmit={step === "request" ? onRequest : onReset}
        className="w-full max-w-sm bg-white rounded-xl shadow p-6 space-y-4"
      >
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-brand-700">{t("auth.forgot_password_title")}</h1>
          <LanguageSwitcher />
        </div>
        <p className="text-sm text-slate-600">{t("auth.forgot_password_subtitle")}</p>

        <label className="block text-sm">
          {t("auth.email")}
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            disabled={step === "reset"}
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500 disabled:bg-slate-50 disabled:text-slate-500"
          />
        </label>

        {step === "reset" && (
          <>
            <label className="block text-sm">
              {t("auth.verify_code_label")}
              <input
                inputMode="numeric"
                autoComplete="one-time-code"
                maxLength={6}
                pattern="\d{6}"
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                required
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-center tracking-[0.4em] text-lg focus:border-brand-500 focus:ring-brand-500"
              />
            </label>

            <label className="block text-sm">
              {t("auth.new_password_label")}
              <input
                type="password"
                minLength={8}
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500"
              />
            </label>
          </>
        )}

        {error && <div className="text-sm text-red-600">{error}</div>}
        {info && step === "reset" && <div className="text-sm text-emerald-600">{info}</div>}

        <button
          type="submit"
          disabled={pending}
          className="w-full rounded-md bg-brand-600 hover:bg-brand-700 text-white py-2 text-sm font-medium disabled:opacity-50"
        >
          {pending
            ? t("common.loading")
            : step === "request"
              ? t("auth.forgot_request_button")
              : t("auth.reset_password_button")}
        </button>

        {step === "reset" && (
          <div className="text-sm text-center">
            <button
              type="button"
              onClick={() => {
                setStep("request");
                setCode("");
                setNewPassword("");
                setError(null);
                setInfo(null);
              }}
              className="text-slate-600 hover:underline"
            >
              {t("common.back")}
            </button>
          </div>
        )}

        <div className="text-sm text-slate-600 text-center">
          <Link to="/login" className="text-brand-700 hover:underline">
            {t("auth.sign_in")}
          </Link>
        </div>
      </form>
    </div>
  );
}
