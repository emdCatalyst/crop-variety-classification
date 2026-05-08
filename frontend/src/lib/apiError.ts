/**
 * FastAPI returns `detail` as either a string (HTTPException) or an array of
 * Pydantic validation objects (`{loc, msg, type, input}`). Rendering an array
 * directly inside JSX throws and crashes the tree, so always coerce to string.
 */
export function apiErrorMessage(err: unknown, fallback: string): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => {
        if (typeof d === "string") return d;
        if (d && typeof d === "object" && typeof (d as { msg?: unknown }).msg === "string") {
          return (d as { msg: string }).msg;
        }
        return null;
      })
      .filter((s): s is string => Boolean(s));
    if (parts.length > 0) return parts.join("; ");
  }
  return fallback;
}
