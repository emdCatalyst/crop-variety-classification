import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Analysis, api } from "@/api/client";

const REQUIRED = 12;

export default function UploadPage() {
  const { t } = useTranslation();
  const [files, setFiles] = useState<File[]>([]);
  const [sourceName, setSourceName] = useState("");
  const [smooth, setSmooth] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const navigate = useNavigate();

  function onPick(ev: React.ChangeEvent<HTMLInputElement>) {
    const list = Array.from(ev.target.files ?? []);
    setFiles(list);
    setError(
      list.length === REQUIRED
        ? null
        : t("upload.select_exact", { count: REQUIRED, got: list.length })
    );
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (files.length !== REQUIRED) {
      setError(t("upload.select_exact", { count: REQUIRED, got: files.length }));
      return;
    }
    setPending(true);
    setError(null);
    try {
      const fd = new FormData();
      files
        .slice()
        .sort((a, b) => a.name.localeCompare(b.name))
        .forEach((f) => fd.append("files", f, f.name));
      fd.append("source_name", sourceName || "uploaded_sequence");
      fd.append("smooth", smooth ? "true" : "false");

      const { data } = await api.post<Analysis>("/analyses", fd);
      navigate(`/analyses/${data.id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail ?? t("upload.upload_failed"));
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">{t("upload.title")}</h1>

      <form onSubmit={onSubmit} className="bg-white rounded-lg shadow-sm p-6 space-y-5">
        <label className="block text-sm">
          {t("upload.source_name")}
          <input
            value={sourceName}
            onChange={(e) => setSourceName(e.target.value)}
            placeholder={t("upload.source_placeholder") ?? ""}
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500"
          />
        </label>

        <div>
          <label className="block text-sm font-medium mb-2">
            {t("upload.files_label", { count: REQUIRED })}
          </label>
          <input
            type="file"
            multiple
            accept=".tif,.tiff,image/tiff"
            onChange={onPick}
            className="block w-full text-sm file:mr-4 file:rounded-md file:border-0 file:bg-brand-600 file:text-white file:px-4 file:py-2 hover:file:bg-brand-700"
          />
          {files.length > 0 && (
            <ul className="mt-3 text-xs text-slate-600 max-h-40 overflow-y-auto bg-slate-50 rounded p-3 space-y-1">
              {files.map((f, i) => (
                <li key={i} className="flex justify-between">
                  <span className="truncate">{f.name}</span>
                  <span className="text-slate-400 ms-2">
                    {(f.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={smooth}
            onChange={(e) => setSmooth(e.target.checked)}
          />
          {t("upload.smooth_label")}
        </label>

        {error && <div className="text-sm text-red-600">{error}</div>}

        <button
          type="submit"
          disabled={pending || files.length !== REQUIRED}
          className="rounded-md bg-brand-600 hover:bg-brand-700 text-white px-5 py-2 text-sm font-medium disabled:opacity-50"
        >
          {pending ? t("upload.uploading") : t("upload.run_inference")}
        </button>
      </form>
    </div>
  );
}
