// Headless smoke test of a viewer page: open it, wait for frames, press keys,
// screenshot, and print the console. Usage:
//   node shot.mjs [--url http://localhost:63317/sim] [--out /tmp/sim.png]
//                 [--keys "Escape,1"] [--wait 6000] [--width 1400 --height 860]
// Needs a lab (uv run duck-lab --world <scenario>) and the viewer (npm run dev)
// already running — see bringup.sh. Playwright is resolved from the global
// node_modules on the web runner; locally `npm i -g playwright` or point
// PLAYWRIGHT_MODULE at an install. CHROMIUM_PATH overrides the binary.
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith("--")) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith("--") ? arr[i + 1] : "true"]);
  return acc;
}, []));
const url = args.url ?? "http://localhost:63317/sim";
const out = args.out ?? "/tmp/sim.png";
const keys = (args.keys ?? "Escape").split(",").filter(Boolean);
const wait = Number(args.wait ?? 6000);
let pw;
for (const cand of [process.env.PLAYWRIGHT_MODULE, "playwright", "/opt/node22/lib/node_modules/playwright"]) {
  if (!cand) continue;
  try { pw = require(cand); break; } catch { /* try next */ }
}
if (!pw) { console.error("playwright not found: npm i -g playwright, or set PLAYWRIGHT_MODULE"); process.exit(2); }
const launch = { args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] };
if (process.env.CHROMIUM_PATH) launch.executablePath = process.env.CHROMIUM_PATH;
else if (process.platform === "linux") launch.executablePath = "/opt/pw-browsers/chromium";
const browser = await pw.chromium.launch(launch);
const page = await browser.newPage({ viewport: { width: Number(args.width ?? 1400), height: Number(args.height ?? 860) } });
const logs = [];
page.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
await page.goto(url, { waitUntil: "networkidle" });
await page.waitForTimeout(wait);
for (const k of keys) { await page.keyboard.press(k); await page.waitForTimeout(400); }
await page.waitForTimeout(1200);
await page.screenshot({ path: out });
const live = await page.locator("text=● live").count();
const noise = /Download the React DevTools|Fast Refresh|\[HMR\]|THREE\.Clock/;
console.log(`screenshot: ${out}\nlive badge: ${live}\n` + logs.filter((l) => !noise.test(l)).slice(0, 20).join("\n"));
await browser.close();
