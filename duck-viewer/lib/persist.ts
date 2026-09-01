// Tiny localStorage JSON helpers, all keys under the "ducklab." prefix.
// Guarded so storage failures (quota, private mode, SSR) never break the UI.

const PREFIX = "ducklab.";

export function loadJSON<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(PREFIX + key);
    if (raw == null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function saveJSON(key: string, value: unknown) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // storage full / blocked — persistence is best-effort
  }
}
