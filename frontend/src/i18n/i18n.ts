import i18n from "i18next";
import LanguageDetector from "i18next-browser-languagedetector";
import { initReactI18next } from "react-i18next";

import en from "./en.json";
import fr from "./fr.json";
import ar from "./ar.json";

export const SUPPORTED_LANGUAGES = ["en", "fr", "ar"] as const;
export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

export const RTL_LANGUAGES = new Set<SupportedLanguage>(["ar"]);

export function applyDocumentDirection(lang: string) {
  const code = (SUPPORTED_LANGUAGES as readonly string[]).includes(lang) ? lang : "en";
  document.documentElement.lang = code;
  document.documentElement.dir = RTL_LANGUAGES.has(code as SupportedLanguage) ? "rtl" : "ltr";
}

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      fr: { translation: fr },
      ar: { translation: ar },
    },
    fallbackLng: "en",
    supportedLngs: SUPPORTED_LANGUAGES as unknown as string[],
    interpolation: { escapeValue: false },
    detection: {
      order: ["localStorage", "cookie", "navigator"],
      caches: ["localStorage", "cookie"],
      lookupLocalStorage: "agrovision_lng",
      lookupCookie: "agrovision_lng",
    },
  });

applyDocumentDirection(i18n.language);
i18n.on("languageChanged", applyDocumentDirection);

export default i18n;
