import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ShieldCheck, User as UserIcon } from "lucide-react";
import { User } from "@/api/client";

export default function ProfileMenu({ user }: { user: User }) {
  const { t } = useTranslation();
  const isAdmin = user.role === "admin";

  return (
    <div className="relative group">
      <Link
        to="/settings"
        className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white pe-3 ps-1 py-1 text-sm hover:border-brand-300 hover:bg-brand-50/40 transition-colors"
        aria-label={user.display_name}
      >
        <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-brand-100 text-brand-700">
          {isAdmin ? (
            <ShieldCheck size={14} aria-hidden />
          ) : (
            <UserIcon size={14} aria-hidden />
          )}
        </span>
        <span className="hidden lg:inline text-slate-700 max-w-[140px] truncate">
          {user.display_name}
        </span>
      </Link>

      <div
        role="tooltip"
        className="absolute end-0 mt-2 w-64 rounded-xl border border-slate-200 bg-white shadow-lg p-4 z-50 opacity-0 invisible -translate-y-1 group-hover:opacity-100 group-hover:visible group-hover:translate-y-0 group-focus-within:opacity-100 group-focus-within:visible group-focus-within:translate-y-0 transition-all duration-150 ease-out"
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-brand-100 text-brand-700 flex items-center justify-center">
            {isAdmin ? (
              <ShieldCheck size={18} aria-hidden />
            ) : (
              <UserIcon size={18} aria-hidden />
            )}
          </div>
          <div className="min-w-0">
            <div className="font-semibold text-slate-900 truncate">
              {user.display_name}
            </div>
            <div className="text-xs text-slate-500 truncate">{user.email}</div>
          </div>
        </div>
        <div className="mt-3 flex items-center gap-2 text-xs">
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded-full font-medium ${
              isAdmin
                ? "bg-amber-50 text-amber-700 border border-amber-200"
                : "bg-slate-100 text-slate-700 border border-slate-200"
            }`}
          >
            {isAdmin ? t("profile.role_admin") : t("profile.role_user")}
          </span>
        </div>
        <p className="text-[11px] text-slate-400 mt-3">
          {t("profile.click_to_settings")}
        </p>
      </div>
    </div>
  );
}
