import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { resendVerification, verifyEmail } from "@/api/auth";
import { User } from "@/api/client";
import LanguageSwitcher from "@/components/LanguageSwitcher";
import { apiErrorMessage } from "@/lib/apiError";

const RESEND_COOLDOWN_S = 30;

export default function VerifyEmailPage({ onSignIn }: { onSignIn: (u: User) => void }) {
  const { t } = useTranslation();
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const email = useMemo(() => params.get("email") ?? "", [params]);
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [cooldown, setCooldown] = useState(0);

  useEffect(() => {
    if (cooldown <= 0) return;
    const id = window.setTimeout(() => setCooldown(cooldown - 1), 1000);
    return () => window.clearTimeout(id);
  }, [cooldown]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setPending(true);
    setError(null);
    setInfo(null);
    try {
      const u = await verifyEmail(email, code);
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

  async function onResend() {
    if (cooldown > 0 || !email) return;
    setError(null);
    setInfo(null);
    try {
      await resendVerification(email);
      setInfo(t("auth.resend_sent"));
      setCooldown(RESEND_COOLDOWN_S);
    } catch (err) {
      setError(apiErrorMessage(err, t("auth.code_invalid")));
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-brand-50 to-white">
      <form onSubmit={onSubmit} className="w-full max-w-sm bg-white rounded-xl shadow p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-brand-700">{t("auth.verify_email_title")}</h1>
          <LanguageSwitcher />
        </div>
        <p className="text-sm text-slate-600">
          {t("auth.verify_email_subtitle")}{" "}
          {email && <span className="font-medium text-slate-900">{email}</span>}
        </p>

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

        {error && <div className="text-sm text-red-600">{error}</div>}
        {info && <div className="text-sm text-emerald-600">{info}</div>}

        <button
          type="submit"
          disabled={pending || code.length !== 6}
          className="w-full rounded-md bg-brand-600 hover:bg-brand-700 text-white py-2 text-sm font-medium disabled:opacity-50"
        >
          {pending ? t("common.loading") : t("auth.verify_button")}
        </button>

        <div className="text-sm text-center">
          <button
            type="button"
            onClick={onResend}
            disabled={cooldown > 0 || !email}
            className="text-brand-700 hover:underline disabled:text-slate-400 disabled:no-underline"
          >
            {cooldown > 0
              ? t("auth.resend_cooldown", { seconds: cooldown })
              : t("auth.resend_code")}
          </button>
        </div>

        <div className="text-sm text-slate-600 text-center">
          <Link to="/login" className="text-brand-700 hover:underline">
            {t("auth.sign_in")}
          </Link>
        </div>
      </form>
    </div>
  );
}
