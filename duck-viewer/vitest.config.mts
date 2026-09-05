import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// Node environment on purpose: the tests that exist are arithmetic — canvas
// rectangles and camera projections — and pull in real three for the camera
// math. Nothing here wants a DOM, and nothing here can substitute for looking
// at the page (.claude/skills/sim-smoke).
export default defineConfig({
  resolve: { alias: { "@": fileURLToPath(new URL(".", import.meta.url)) } },
  test: { include: ["{lib,components,app}/**/*.test.{ts,tsx}"] },
});
