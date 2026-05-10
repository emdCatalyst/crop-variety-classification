import { useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import LanguageSwitcher from "@/components/LanguageSwitcher";

type Expert = {
  name: string;
  email: string;
  image: string;
};

const EXPERTS: Expert[] = [
  {
    name: "Dr. Jane Doe",
    email: "jane.doe@example.com",
    image: "/portraits/expert1.jpg",
  },
  {
    name: "Dr. John Smith",
    email: "john.smith@example.com",
    image: "/portraits/expert2.jpg",
  },
];

export default function HomePage() {
  const { t } = useTranslation();

  return (
    <div className="min-h-screen bg-gradient-to-b from-brand-50 via-white to-white">
      <header className="container flex items-center justify-between h-14">
        <span className="font-bold text-brand-700 text-lg">{t("app.name")}</span>
        <div className="flex items-center gap-2">
          <LanguageSwitcher />
          <Link
            to="/login"
            className="text-sm rounded-md px-3 py-1.5 text-slate-700 hover:bg-slate-100"
          >
            {t("auth.sign_in")}
          </Link>
          <Link
            to="/signup"
            className="text-sm rounded-md px-3 py-1.5 bg-brand-600 hover:bg-brand-700 text-white font-medium"
          >
            {t("home.get_started")}
          </Link>
        </div>
      </header>

      <main className="container">
        <section className="py-16 md:py-24 grid gap-10 md:grid-cols-2 items-center">
          <div className="space-y-5">
            <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 leading-tight">
              {t("home.headline")}
            </h1>
            <p className="text-lg text-slate-600">{t("home.subhead")}</p>
            <div className="flex flex-wrap gap-3">
              <Link
                to="/signup"
                className="rounded-md bg-brand-600 hover:bg-brand-700 text-white px-5 py-2.5 text-sm font-semibold"
              >
                {t("home.get_started")}
              </Link>
              <Link
                to="/login"
                className="rounded-md border border-slate-300 hover:bg-slate-50 px-5 py-2.5 text-sm font-semibold"
              >
                {t("auth.sign_in")}
              </Link>
            </div>
          </div>

          <div className="relative aspect-square md:aspect-[4/5] bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-200">
            <div className="absolute inset-0 bg-gradient-to-br from-brand-100 via-white to-brand-50" />
            <div className="absolute inset-6 grid grid-cols-3 gap-2">
              {Array.from({ length: 12 }).map((_, i) => (
                <div
                  key={i}
                  className="rounded-md"
                  style={{
                    background: `linear-gradient(135deg, hsl(${
                      (i * 27) % 360
                    }, 60%, 70%), hsl(${(i * 27 + 60) % 360}, 60%, 55%))`,
                    opacity: 0.85,
                  }}
                />
              ))}
            </div>
            <div className="absolute bottom-4 inset-x-4 text-xs text-slate-500 text-center">
              {t("home.preview_caption")}
            </div>
          </div>
        </section>

        <section className="py-12 grid gap-6 md:grid-cols-3">
          <Feature
            title={t("home.feature_inference_title")}
            body={t("home.feature_inference_body")}
          />
          <Feature
            title={t("home.feature_health_title")}
            body={t("home.feature_health_body")}
          />
          <Feature
            title={t("home.feature_reports_title")}
            body={t("home.feature_reports_body")}
          />
        </section>

        <section className="py-12">
          <div className="text-center mb-8">
            <h2 className="text-2xl md:text-3xl font-bold text-slate-900">
              {t("home.backed_by_title")}
            </h2>
            <p className="text-sm text-slate-600 mt-2">
              {t("home.backed_by_subtitle")}
            </p>
          </div>
          <div className="grid gap-6 sm:grid-cols-2 max-w-2xl mx-auto">
            {EXPERTS.map((expert) => (
              <ExpertCard key={expert.email} expert={expert} />
            ))}
          </div>
        </section>

        <footer className="py-10 text-center text-xs text-slate-500">
          {t("home.footer", { year: new Date().getFullYear() })}
        </footer>
      </main>
    </div>
  );
}

function Feature({ title, body }: { title: string; body: string }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-5">
      <h3 className="font-semibold text-slate-900 mb-2">{title}</h3>
      <p className="text-sm text-slate-600">{body}</p>
    </div>
  );
}

function ExpertCard({ expert }: { expert: Expert }) {
  const { t } = useTranslation();
  const [imageFailed, setImageFailed] = useState(false);
  const initials = expert.name
    .replace(/^(Dr|Mr|Mrs|Ms|Prof)\.?\s+/i, "")
    .split(/\s+/)
    .map((part) => part[0])
    .filter(Boolean)
    .slice(0, 2)
    .join("")
    .toUpperCase();

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-100 p-6 flex flex-col items-center text-center">
      <div className="w-24 h-24 rounded-full overflow-hidden bg-brand-100 ring-2 ring-brand-200 mb-4 flex items-center justify-center">
        {imageFailed ? (
          <span className="text-2xl font-semibold text-brand-700">
            {initials}
          </span>
        ) : (
          <img
            src={expert.image}
            alt={expert.name}
            className="w-full h-full object-cover"
            onError={() => setImageFailed(true)}
          />
        )}
      </div>
      <h3 className="font-semibold text-slate-900">{expert.name}</h3>
      <a
        href={`mailto:${expert.email}`}
        aria-label={t("home.backed_by_email_aria", { name: expert.name })}
        className="text-sm text-brand-700 hover:text-brand-800 hover:underline mt-1 break-all"
      >
        {expert.email}
      </a>
    </div>
  );
}
