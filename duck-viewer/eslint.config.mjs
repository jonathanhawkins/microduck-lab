import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // The viewer reads shared mutable stores (refs) inside the r3f render
  // loop ON PURPOSE — per-frame data (duck poses, HUD numbers) arrives over
  // WebSocket at 30-60 Hz, and subscribing components via useSyncExternalStore
  // re-rendered the whole tree every frame (measured stalls; see
  // duck-viewer/README.md). react-hooks/refs flags exactly that pattern, so
  // it is off here; the rest of the hooks rules stay on.
  {
    rules: {
      "react-hooks/refs": "off",
      // Same architecture, same verdict: the React Compiler's purity
      // and immutability rules reject per-frame mutation of shared
      // stores and Date.now()-driven timer UI — both deliberate here.
      "react-hooks/immutability": "off",
      "react-hooks/purity": "off",
    },
  },
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
