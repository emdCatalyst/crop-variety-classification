import { useTranslation } from "react-i18next";
import { SUPPORTED_LANGUAGES, SupportedLanguage } from "@/i18n/i18n";
import { updateProfile } from "@/api/settings";

export default function LanguageSwitcher({
  authenticated = false,
  onLanguageChange,
}: {
  authenticated?: boolean;
  onLanguageChange?: (lang: SupportedLanguage) => void;
}) {
  const { i18n, t } = useTranslation();

  async function onChange(ev: React.ChangeEvent<HTMLSelectElement>) {
    const next = ev.target.value as SupportedLanguage;
    await i18n.changeLanguage(next);
    onLanguageChange?.(next);
    if (authenticated) {
      try {
        await updateProfile({ language: next });
      } catch {
        // best-effort; the change still persists locally via localStorage
      }
    }
  }

  return (
    <label className="text-sm text-slate-600 inline-flex items-center gap-2">
      <span className="sr-only">{t("common.language")}</span>
      <select
        value={i18n.language.split("-")[0]}
        onChange={onChange}
        className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm focus:border-brand-500 focus:ring-brand-500"
      >
        {SUPPORTED_LANGUAGES.map((lng) => (
          <option key={lng} value={lng}>
            {labelFor(lng)}
          </option>
        ))}
      </select>
    </label>
  );
}

function labelFor(code: SupportedLanguage): string {
  switch (code) {
    case "en":
      return "English";
    case "fr":
      return "Français";
    case "ar":
      return "العربية";
  }
}
