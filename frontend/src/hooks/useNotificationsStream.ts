import { useEffect } from "react";

const STREAM_URL = "/api/v1/notifications/stream";

export function useNotificationsStream(onNotification: () => void) {
  useEffect(() => {
    let es: EventSource | null = null;
    try {
      es = new EventSource(STREAM_URL, { withCredentials: true });
      es.addEventListener("notification", () => onNotification());
      es.onerror = () => {
        // browser auto-retries
      };
    } catch {
      // ignore — caller can fall back to polling
    }
    return () => es?.close();
  }, [onNotification]);
}
