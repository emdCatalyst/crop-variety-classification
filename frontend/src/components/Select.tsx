import { ReactNode } from "react";
import * as RS from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";

export type SelectOption<T extends string = string> = {
  value: T;
  label: ReactNode;
};

export function Select<T extends string = string>({
  value,
  onValueChange,
  options,
  placeholder,
  disabled,
  triggerClassName = "",
  ariaLabel,
}: {
  value: T;
  onValueChange: (value: T) => void;
  options: SelectOption<T>[];
  placeholder?: string;
  disabled?: boolean;
  triggerClassName?: string;
  ariaLabel?: string;
}) {
  return (
    <RS.Root
      value={value}
      onValueChange={(v) => onValueChange(v as T)}
      disabled={disabled}
    >
      <RS.Trigger
        aria-label={ariaLabel}
        className={`group inline-flex items-center justify-between gap-2 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 hover:border-slate-400 focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/30 disabled:opacity-50 disabled:cursor-not-allowed ${triggerClassName}`}
      >
        <RS.Value placeholder={placeholder} />
        <RS.Icon className="text-slate-400 group-data-[state=open]:rotate-180 transition-transform">
          <ChevronDown size={14} aria-hidden />
        </RS.Icon>
      </RS.Trigger>
      <RS.Portal>
        <RS.Content
          position="popper"
          sideOffset={4}
          className="z-50 min-w-[var(--radix-select-trigger-width)] rounded-md border border-slate-200 bg-white shadow-lg overflow-hidden animate-slide-down"
        >
          <RS.Viewport className="p-1">
            {options.map((opt) => (
              <RS.Item
                key={opt.value}
                value={opt.value}
                className="flex items-center gap-2 px-2 py-1.5 text-sm rounded-md text-slate-700 cursor-pointer outline-none data-[highlighted]:bg-brand-50 data-[highlighted]:text-brand-800 data-[state=checked]:font-semibold"
              >
                <span className="w-4 inline-flex justify-center">
                  <RS.ItemIndicator>
                    <Check size={14} className="text-brand-600" aria-hidden />
                  </RS.ItemIndicator>
                </span>
                <RS.ItemText>{opt.label}</RS.ItemText>
              </RS.Item>
            ))}
          </RS.Viewport>
        </RS.Content>
      </RS.Portal>
    </RS.Root>
  );
}
