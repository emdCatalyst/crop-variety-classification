/**
 * Two-tone notification chime via Web Audio API. No asset dependency, fails
 * silently if the browser blocks audio (autoplay policy, missing API, etc.).
 */
let cachedCtx: AudioContext | null = null;

function getCtx(): AudioContext | null {
  try {
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    if (!cachedCtx) cachedCtx = new Ctor();
    if (cachedCtx.state === "suspended") {
      cachedCtx.resume().catch(() => {});
    }
    return cachedCtx;
  } catch {
    return null;
  }
}

export function playNotificationChime(): void {
  const ctx = getCtx();
  if (!ctx) return;
  try {
    const now = ctx.currentTime;
    [
      { freq: 880, at: 0 },
      { freq: 660, at: 0.13 },
    ].forEach(({ freq, at }) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, now + at);
      gain.gain.setValueAtTime(0, now + at);
      gain.gain.linearRampToValueAtTime(0.18, now + at + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, now + at + 0.22);
      osc.connect(gain).connect(ctx.destination);
      osc.start(now + at);
      osc.stop(now + at + 0.24);
    });
  } catch {
    // best-effort
  }
}
