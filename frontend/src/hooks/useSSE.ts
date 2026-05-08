import { useEffect, useState } from "react";

export type SSEEvent = {
  stage: string;
  message?: string;
  result_id?: number;
};

export function useAnalysisEvents(analysisId: number | null, enabled: boolean) {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [stage, setStage] = useState<string>("queued");

  useEffect(() => {
    if (!enabled || !analysisId) return;
    const url = `/api/v1/analyses/${analysisId}/events`;
    const es = new EventSource(url, { withCredentials: true });

    const onMessage = (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data) as SSEEvent;
        if (data.stage) setStage(data.stage);
        if (data.message) setEvents((prev) => [...prev, data]);
      } catch {
        /* ignore */
      }
    };

    ["status", "loading", "inferring", "rendering", "done", "failed"].forEach((name) => {
      es.addEventListener(name, onMessage as EventListener);
    });

    es.onerror = () => {
      es.close();
    };

    return () => es.close();
  }, [analysisId, enabled]);

  return { events, stage };
}
