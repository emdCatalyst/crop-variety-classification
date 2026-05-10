import { useTranslation } from "react-i18next";
import { SUPPORTED_LANGUAGES, SupportedLanguage } from "@/i18n/i18n";
import { updateProfile } from "@/api/settings";
import { Select } from "./Select";

export default function LanguageSwitcher({
  authenticated = false,
  onLanguageChange,
}: {
  authenticated?: boolean;
  onLanguageChange?: (lang: SupportedLanguage) => void;
}) {
  const { i18n, t } = useTranslation();

  async function onChange(next: SupportedLanguage) {
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
    <Select<SupportedLanguage>
      value={(i18n.language.split("-")[0] as SupportedLanguage) ?? "en"}
      onValueChange={onChange}
      ariaLabel={t("common.language") ?? undefined}
      options={SUPPORTED_LANGUAGES.map((lng) => ({
        value: lng,
        label: (
          <span className="inline-flex items-center gap-2">
            <span aria-hidden className="text-base leading-none">
              {flagFor(lng)}
            </span>
            <span>{labelFor(lng)}</span>
          </span>
        ),
      }))}
    />
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

function flagFor(code: SupportedLanguage): string {
  switch (code) {
    case "en":
      return "🇬🇧";
    case "fr":
      return "🇫🇷";
    case "ar":
      return "🇸🇦";
  }
}
