// Types + client for the duck-lab server (microduck_local viz_server.py).

// `?lab=host:port` points the page at a different lab (a scratch server, a
// lab on another machine). The module only loads client-side (the viewer is
// ssr:false), but guard anyway so an SSR import can't crash the build.
const LAB_HOST =
  (typeof window !== "undefined" &&
    new URLSearchParams(window.location.search).get("lab")) ||
  "127.0.0.1:8788";
export const LAB_HTTP = `http://${LAB_HOST}`;
export const LAB_WS = `ws://${LAB_HOST}/ws`;

export interface SceneMesh {
  v: number[]; // flat xyz, MuJoCo frame (Z-up)
  f: number[]; // flat triangle indices
}
export interface SceneGeom {
  mesh: number;
  body: number;
  pos: [number, number, number];
  quat: [number, number, number, number]; // MuJoCo wxyz
  /** MJCF material name ("" = none) — key for client-side color fixes. */
  mat?: string;
  /** Material rgba from the compiled model, sRGB floats. Absent on servers
   *  that predate color streaming — the viewer then falls back to its old
   *  per-body palette. */
  rgba?: [number, number, number, number];
  /** MJCF mesh asset name (e.g. head_link) — G1 visor vs body materials. */
  name?: string;
}
export interface Scene {
  bodies: string[];
  meshes: SceneMesh[];
  geoms: SceneGeom[];
  /** Multiply mesh vertices by this to get metres. The duck's scene streams
   *  metres and omits it; the G1's streams MILLIMETRE ints (its mesh dump is
   *  ~21 MB that way instead of ~78 MB of floats) and sets 0.001. */
  vertScale?: number;
}

export interface DuckFrame {
  id: string; // stable identity ("d0".."dN", "trainee", "helper1"…) — survives renames
  name: string; // mutable display label (tracks the assigned policy)
  /** Brain provenance — the palette id this duck runs ("run:<name>",
   *  "ckpt:<name>@Nk", "pollen:<name>") or null (a zero-infer trainee before
   *  its first snapshot, or a server predating the field). Lets selection
   *  load the duck's run into the teach panel. */
  policy?: string | null;
  /** Which body to draw for this row — any registry id ("microduck", "g1",
   *  "mars", "menagerie:unitree_go2"); the meshes come from
   *  GET /scene?robot=<id>. Absent on servers that predate robot selection
   *  — those rosters are all ducks. */
  robot?: RobotId;
  falls: number;
  step: number;
  rew: number;
  /** The mouth servo as an opening fraction, 0 shut to 1 wide — the /sim
   *  stream carries it (world/compose.py `split_jaw`); the lab page's ducks
   *  have no mouth joint and leave it absent. Duck.tsx opens the bill to
   *  the wider of this and the duck's voice (lib/mouth.ts). */
  mouth?: number;
  /** Forward speed in m/s, in the duck's HEADING frame, averaged over the
   *  last ~0.5 s of control steps (server: Duck.forward_speed). null for the
   *  one frame after an episode reset, before the window has a sample. */
  speed?: number | null;
  /** The forward speed the duck is being ASKED for (twist_cmd[0]), so the
   *  achieved figure can be read against it — policies here deliver about
   *  half what they are commanded. null for trick policies, which run a
   *  pinned-zero command and have nothing to compare against. */
  cmdSpeed?: number | null;
  steerable?: boolean; // false = trick policy, ignores drive commands
  // How the current practice episode STARTED ("standing", "mid-roll 265°",
  // "landed", "inverted"…) — only trick-behavior envs report it. Shown under
  // the duck's label so rehearsal spawns that look like ordinary standing
  // are still legible.
  spawn?: string | null;
  /** True while a showcase duck's demo spotter is assisting (see the server's
   *  spotter_fn) — shown in-scene so an assisted stretch is never mistaken
   *  for the policy doing it unaided. */
  assist?: boolean;
  /** A showcase duck hot-swaps to a standing brain once the trick finishes
   *  (the robot's own pattern) — `handed` is true once control moved over,
   *  `handoff` names the brain taking it. */
  handed?: boolean;
  handoff?: string | null;
  /** The ball a 🔎 find_ball duck is looking for — [x, y, z, radius] in the
   *  duck's own world frame, like `bodies`. It lives in the env, not the
   *  physics, so it is streamed here rather than as a body. null/absent for
   *  every other brain. */
  ball?: number[] | null;
  /** Where this slot stands on the lab floor, MuJoCo XY metres. The SERVER
   *  lays the grid out (viz_server.lab_slot_offsets) because the pitch is a
   *  property of the robot — RobotSpec.lab_spacing_m, 0.65 m for a 25 cm duck
   *  and 1.88 m for a 1.3 m G1 — and a roster can hold both. Absent on a
   *  server that predates it; the viewer then falls back to its own
   *  duck-pitched grid. */
  offset?: [number, number];
  bodies: number[][]; // per body: [x, y, z, qw, qx, qy, qz]
}

/** ~1 Hz psutil sample the server folds into every frame. */
export interface ProcStats {
  cpu: number; // % of one core (can exceed 100 for multi-process trees)
  memMb: number;
}
export interface SystemStats {
  cpu: number; // machine-wide, 0-100
  mem: number; // machine-wide, 0-100
  lab: ProcStats;
  trainer: ProcStats | null; // null when no trainer subprocess is alive
  trainFps: number | null; // training steps/s from progress.jsonl, null right after a restart
}

/** One term of a behavior's reward recipe (weights are the DEFAULTS). */
export interface TermCard {
  key: string;
  friendly: string;
  weight: number;
  isPenalty: boolean;
}
export interface BehaviorCard {
  id: string;
  emoji: string;
  title: string;
  /** Authored motion clip used by imitation runs. The server includes this
   *  when the display title is specialized to `Perform “…”`. */
  clip?: string;
  description: string;
  howItLearns: string;
  successMetric: string;
  defaultSteps: number;
  /** Staged curriculum in training order — each stage a separate run
   *  fine-tuned from the previous. Empty/absent = one single run.
   *  `detail` is the stage's plain-English story (what it rehearses). */
  curriculum?: { label: string; steps: number; detail?: string }[];
  terms: TermCard[];
  /** Catalog terms NOT in this recipe (weights are their defaults). Adding
   *  one = POSTing its key in the /teach weights dict; the next run then
   *  owns it (moves into `terms`, drops out of here). */
  availableTerms?: TermCard[];
}
export interface TrainingProgress {
  steps?: number;
  total?: number;
  ep_rew?: number;
  ep_len?: number;
  terms?: Record<string, number>;
  snapshots?: number;
  elapsed_s?: number;
  done?: boolean;
  /** Cumulative across a staged curriculum (== steps/total for single-run
   *  jobs) — what long-lived counters should show, so progress never appears
   *  to reset when a stage hands off. */
  overallSteps?: number;
  overallTotal?: number;
  /** Wall-clock seconds the JOB has been training — spans stage handoffs
   *  and warm restarts, where elapsed_s starts over with each subprocess.
   *  Frozen at finish; null for adopted (already-finished) runs. */
  overallElapsed?: number | null;
}
/** Active stage of a staged-curriculum teach job (1-based idx). */
export interface TrainingStage {
  idx: number;
  count: number;
  label: string;
  /** Plain-English story of what the stage rehearses (the inspector text). */
  detail?: string;
  /** 1-based stage the chain actually began at (startStage) — stages before
   *  it were skipped, warm-started from an earlier run. Absent = 1. */
  start?: number;
}
export interface TrainingPayload {
  runName: string;
  status: "training" | "done" | "stopped" | "failed";
  behavior: BehaviorCard;
  progress: TrainingProgress;
  /** null/absent = single-run job; set while a curriculum chain trains. */
  stage?: TrainingStage | null;
  /** termKey → weight the trainer actually uses (defaults + slider overrides). */
  weights: Record<string, number>;
  /** Per-stage OVERRIDES only, keyed by 1-based stage index as a string —
   *  layer them over `weights` (stage wins per key) for a stage's merged
   *  sliders. Empty/absent for single-run jobs. */
  stageWeights?: Record<string, Record<string, number>>;
  /** Practice budget IN FORCE, in steps: per-stage counts in stage order
   *  (a single entry for a single-run job) and their sum. The panel shows
   *  these rather than the recipe's declared numbers, so a run can never
   *  advertise a budget it isn't training under. */
  stageSteps?: number[];
  stepBudget?: number;
  /** The TOTAL the user chose, or null while the recipe's own plan is in
   *  force — stepBudget already has per-stage pins folded in, so previewing
   *  an edit re-splits around THIS. */
  chosenBudget?: number | null;
  /** The explicitly PINNED subset of stageSteps (1-based string keys) — what
   *  the user set by hand; every other stage takes its proportional share. */
  stageBudgets?: Record<string, number>;
  envs: number; // parallel training environments (10 base + 2 per helper)
  helpers: number;
  maxHelpers?: number; // hard cap (server DUCK_MAX_HELPERS; warns past the CPU sweet spot)
  restarting: boolean; // true while a helper spawn/remove warm-restarts the trainer
}

export interface Frame {
  cmd: [number, number, number];
  mode: "auto" | "manual";
  ducks: DuckFrame[];
  events?: string[]; // one-shot toast lines (each appears in a single frame)
  stats?: SystemStats;
  training?: TrainingPayload | null;
}

export interface Policy {
  id: string; // e.g. "pollen:alpha_stand"
  label: string; // e.g. "alpha_stand"
  /** Which palette section this chip belongs to. "runs" and "checkpoints"
   *  are ours; every other value is a SHIPPED group a body declared
   *  ("pollen", "g1") and is described by `Lab.shipped`. A string, not a
   *  union, for the same reason as `RobotId`. */
  group: string;
  path: string;
  /** Newest-artifact timestamp, epoch SECONDS (run policies only) — the
   *  server sorts the "runs" group newest-first by it; the panel renders it
   *  as a relative "2h ago" label. */
  mtime?: number;
  /** Curriculum-chain grouping (run policies named teach-…-sN): the chain
   *  prefix without the -sN suffix, plus the 1-based stage — the panel folds
   *  a chain's stages into one family row of compact chips. */
  chain?: string;
  stage?: number;
  /** Bytes the run dir occupies (run policies only) — shown in the delete
   *  confirmation so the user can see what a delete actually frees. */
  sizeBytes?: number;
  /** Which body this brain drives (server: run.json). Absent on servers that
   *  predate robot selection — everything there is a duck. */
  robot?: RobotId;
  /** One measured sentence about what this policy does, shown in the chip's
   *  tooltip (server: robots/g1._SHIPPED_NOTES for the shipped G1 brains,
   *  a run's record.json for our own runs). */
  note?: string;
  /** Human name for a run — e.g. "Front kick (G1) · perform “g1-front-kick”"
   *  (server: the run's record.json, written by the trainer and by
   *  `describe-run`). Absent on runs with no record, which is why nothing
   *  reads it directly: every chip goes through `policyTitle`. */
  title?: string;
  /** True on the ONE stage of a curriculum chain that MEASURED best. It is
   *  usually NOT the last stage — later stages are kept for comparison after
   *  an earlier one won — so the chain's chip asks `chainPick` rather than
   *  taking the tail. At most one stage of a chain carries it; a chain whose
   *  stages were never compared carries none. */
  pick?: boolean;
  /** The recipe this run practised, as a behaviors id ("backflip",
   *  "g1_front_kick") — what "Our runs" groups by and what a 🎓 chip looks its
   *  best run up under (lib/tricks.ts). Absent on a run the lab could not
   *  place, and on every run of a lab that predates the field. */
  trick?: string;
}

/** What a trick id is called in a heading (GET /policies `tricks`). */
export interface TrickName {
  title: string;
  emoji?: string;
}

/** The chip prefix that says a policy is not a duck. Empty for the duck, so
 *  a one-robot lab's palette reads exactly as it always has — and empty
 *  inside a section that is already one robot ("Unitree G1 (shipped)"), where
 *  repeating it on every chip is noise. */
export function robotTag(robot?: RobotId | string, group?: string): string {
  // A SHIPPED section is one body's by construction (a body declares its own
  // group), so the prefix is noise there whichever body it is — this used to
  // name the G1's group, which a third body's section would have missed.
  if (group && !OUR_GROUPS.includes(group)) return "";
  return !robot || robot === "microduck" ? "" : `${robot} · `;
}

/** What a chip SHOWS for a policy: the run's human title when the server has
 *  a record for it, else the directory name — with the trainer's `teach-`
 *  prefix stripped the way the chain header has always stripped it, so a
 *  chip and the header above it can't read differently. The raw `label`/`id`
 *  is untouched by this and stays what travels: it is the identifier
 *  `--init-from` and the docs need, so the chips keep it in their tooltip. */
export function policyTitle(p: Policy): string {
  return (p.title ?? p.label).replace(/^teach-/, "");
}

/** The stage of a curriculum chain that stands for the whole trick: the one
 *  the record marks as measured best, else the last. The tail is only the
 *  default — a chain often keeps its later stages for comparison after an
 *  earlier one measured better, and handing those out silently would assign
 *  a brain nobody picked. */
export function chainPick(stages: Policy[]): Policy {
  return stages.find((s) => s.pick) ?? stages[stages.length - 1];
}

/** Permanently delete a training run's directory — its policy, checkpoints
 *  and progress log. `chain` treats `name` as a curriculum-chain prefix and
 *  deletes every stage of it in one all-or-nothing call. There is no undo:
 *  only call this behind an explicit user confirmation. Throws with the
 *  server's own message (a run of the ACTIVE training job is refused). */
/** ⚙ BYOK Hugging Face settings. The token goes UP once and never comes
 *  back: the lab validates it against whoami(), stores it 0600 beside runs/,
 *  and every read returns only a mask + the username. It unlocks the real
 *  training step — GPU runs of microduck_rl on HF Jobs, on the user's own
 *  account and dime. */
export interface HfSettings {
  configured: boolean;
  username?: string;
  masked?: string;
}

async function hfError(res: Response, fallback: string): Promise<never> {
  const detail = await res
    .json()
    .then((d: { detail?: string }) => d?.detail)
    .catch(() => undefined);
  throw new Error(detail || fallback);
}

export async function fetchHfSettings(): Promise<HfSettings> {
  const res = await fetch(`${LAB_HTTP}/settings/hf`).catch(() => {
    throw new Error(`can't reach the lab at ${LAB_HOST}`);
  });
  if (!res.ok) return hfError(res, `settings failed: ${res.status}`);
  return res.json();
}

export async function saveHfToken(token: string): Promise<HfSettings> {
  const res = await fetch(`${LAB_HTTP}/settings/hf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  }).catch(() => {
    throw new Error(`can't reach the lab at ${LAB_HOST} — token not saved`);
  });
  if (!res.ok) return hfError(res, `token rejected: ${res.status}`);
  return res.json();
}

export async function deleteHfToken(): Promise<HfSettings> {
  const res = await fetch(`${LAB_HTTP}/settings/hf`, { method: "DELETE" }).catch(() => {
    throw new Error(`can't reach the lab at ${LAB_HOST}`);
  });
  if (!res.ok) return hfError(res, `disconnect failed: ${res.status}`);
  return res.json();
}

export async function deleteRun(
  name: string,
  chain = false
): Promise<{ deleted: string[]; freedBytes: number }> {
  const res = await fetch(
    `${LAB_HTTP}/runs/${encodeURIComponent(name)}${chain ? "?chain=true" : ""}`,
    { method: "DELETE" }
  ).catch(() => {
    // A dead lab throws a bare "Failed to fetch"; say WHERE nothing answered,
    // matching the panel's "can't load policies from :8788" line.
    throw new Error(`can't reach the lab at ${LAB_HOST} — nothing was deleted`);
  });
  if (!res.ok) {
    // FastAPI puts the human-readable reason in `detail`; fall back to the
    // status so a proxy/500 with no JSON body still says something useful.
    const detail = await res
      .json()
      .then((d: { detail?: string }) => d?.detail)
      .catch(() => undefined);
    throw new Error(detail || `delete failed: ${res.status}`);
  }
  return res.json();
}

/** "240 MB" / "1.4 GB" — delete-confirmation sizing, not an exact accounting. */
export function formatBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} KB`;
  return `${n} B`;
}

/** Which body a roster slot (or a palette entry) is — any id the lab's
 *  registry knows (`robots/registry.ids()`): "microduck", "g1", "mars", or a
 *  discovered one like "menagerie:unitree_go2".
 *
 *  A UNION of two literals until Phase 1b, which is what made every third
 *  body a pass over this file and six components. It is a string because the
 *  set lives on the server: a body can arrive from a pip-installed plugin or
 *  from a Menagerie download, and TypeScript cannot know either. Kept as a
 *  named alias rather than inlined so the places that mean "a robot id" still
 *  say so. */
export type RobotId = string;

export async function fetchScene(robot: RobotId = "microduck"): Promise<Scene> {
  const q = robot && robot !== "microduck" ? `?robot=${encodeURIComponent(robot)}` : "";
  const res = await fetch(`${LAB_HTTP}/scene${q}`);
  if (!res.ok) throw new Error(`scene fetch failed: ${res.status}`);
  return res.json();
}

/** Every robot present in a roster frame, deduped. The viewer loads one mesh
 *  set per robot on the stage rather than one per duck. */
export function robotsInFrame(ducks: { robot?: string }[]): RobotId[] {
  const out = new Set<RobotId>();
  for (const d of ducks) out.add(((d.robot as RobotId) || "microduck"));
  return [...out];
}

/** True for palette ids with a training run behind them — the only ones the
 *  teach panel can load (shipped Pollen policies have no recipe to refine). */
export function isRunPolicy(policyId: string | null | undefined): boolean {
  return !!policyId && (policyId.startsWith("run:") || policyId.startsWith("ckpt:"));
}

/** The run-dir name behind a palette id ("run:x" / "ckpt:x@123k" → "x"). */
export function runNameOfPolicy(policyId: string): string {
  return policyId.split(":", 2)[1]?.split("@", 1)[0] ?? policyId;
}

/** POST /teach/load — seat a finished run in the teach panel (no training
 *  started): its recipe streams back in "done" state, sliders unlocked,
 *  fine-tune targeting that run. Refused while a job is actively training. */
export async function loadTeachRun(
  policyId: string
): Promise<{ ok: boolean; message?: string }> {
  const res = await fetch(`${LAB_HTTP}/teach/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ policy: policyId }),
  });
  if (!res.ok) throw new Error(`teach/load failed: ${res.status}`);
  return res.json();
}

/** A body the lab can load (GET /policies `robots`). `ready` is false until
 *  its assets are on disk; `setup` is then the command that fetches them. */
export interface LabRobot {
  id: RobotId;
  label: string;
  /** What a sentence — and every robot switch — calls it ("G1", "MARS"). */
  noun?: string;
  /** What SHAPE of robot this is: "legged" | "wheeled" | "generic". Lets the
   *  UI pick an emoji and a verb ("teach the duck a trick" / "teach MARS a
   *  task") without a table of ids. Absent on a lab that predates it. */
  kind?: string;
  ready: boolean;
  setup?: string;
  /** A POST /robots/{id}/fetch download is running / last one failed. */
  fetching?: boolean;
  fetchError?: string;
  /** Can the 🎬 pose editor open this body? A CAPABILITY, not a kind: the
   *  editor is a walker's tool (effectors, soles, a base link), and the
   *  server answers it by asking the body (lab/robots._animates). Absent on
   *  an older lab, where the panel's own list was the filter. */
  animate?: boolean;
}

/** One shipped section of the palette (GET /policies `shipped`): the `group`
 *  key its chips carry, the heading to show, and which body it belongs to.
 *  The viewer used to hold this list itself, so a third body would have been
 *  a fourth literal in this file. */
export interface ShippedGroup {
  key: string;
  title: string;
  robot: RobotId;
}

/** Start downloading a robot's assets on the lab (the G1 ~140 MB, MARS
 *  7.2 MB, a Menagerie model its own size). Returns at once; poll fetchLab()
 *  until the robot reports `ready`. */
export async function fetchRobot(id: RobotId): Promise<void> {
  const res = await fetch(`${LAB_HTTP}/robots/${encodeURIComponent(id)}/fetch`, { method: "POST" });
  if (!res.ok) throw new Error(`robot fetch failed: ${res.status}`);
}

/** The robots an older lab (no `robots` field) implies: the duck only. */
export const DEFAULT_ROBOTS: LabRobot[] = [
  { id: "microduck", label: "Microduck", noun: "duck", kind: "legged", ready: true, animate: true },
];

/** Which robot a policy drives — an entry without `robot` is the duck's. */
export function policyRobotId(p: Pick<Policy, "robot">): string {
  return p.robot || "microduck";
}

export interface Lab {
  policies: Policy[];
  robots: LabRobot[];
  /** The palette's shipped sections, in registry order. Empty from an older
   *  lab, and `shippedGroups` then derives them from the chips themselves. */
  shipped: ShippedGroup[];
  /** Names for the trick ids the policies carry; empty from an older lab. */
  tricks: Record<string, TrickName>;
}

export async function fetchLab(): Promise<Lab> {
  const res = await fetch(`${LAB_HTTP}/policies`);
  if (!res.ok) throw new Error(`policies fetch failed: ${res.status}`);
  const data: {
    policies: Policy[];
    robots?: LabRobot[];
    shipped?: ShippedGroup[];
    tricks?: Record<string, TrickName>;
  } = await res.json();
  const policies = dedupePolicies(data.policies);
  return {
    policies,
    robots: data.robots?.length ? data.robots : DEFAULT_ROBOTS,
    shipped: data.shipped?.length ? data.shipped : shippedGroups(policies),
    tricks: data.tricks ?? {},
  };
}

/** The shipped sections implied by a chip list, for a lab too old to send
 *  them: every group that is neither ours ("runs", "checkpoints") in the
 *  order it first appears. The heading is the group key itself, which is
 *  what those labs' own palettes fell back to for an unknown group. */
export function shippedGroups(policies: Policy[]): ShippedGroup[] {
  const out: ShippedGroup[] = [];
  for (const p of policies) {
    if (OUR_GROUPS.includes(p.group) || out.some((g) => g.key === p.group)) continue;
    out.push({ key: p.group, title: `${p.group} (shipped)`, robot: policyRobotId(p) });
  }
  return out;
}

/** The palette sections that are OURS rather than a body's shipped drop.
 *  Everything else on a chip's `group` names a shipped section. */
export const OUR_GROUPS = ["runs", "checkpoints"];

export async function fetchPolicies(): Promise<Policy[]> {
  return (await fetchLab()).policies;
}

function dedupePolicies(policies: Policy[]): Policy[] {
  // Keep the first of each id: the server can emit duplicates (two checkpoints
  // of one run in the same 1k-step bucket share a "run@Nk" id), and duplicate
  // ids would collide as React keys in the palette. Either chip sends the same
  // assign/spawn message anyway, so dropping repeats loses nothing.
  const seen = new Set<string>();
  return policies.filter((p) => {
    if (seen.has(p.id)) return false;
    seen.add(p.id);
    return true;
  });
}

/** React keys for a duck-roster render, aligned by index. Normally just the
 *  stable duck id — but the stream is not guaranteed duplicate-free (legacy
 *  lab-state.json rosters used policy labels as duck ids, and the server
 *  restores saved ids verbatim), and two rows sharing a key made React log an
 *  error on every HUD poll and free to drop/duplicate rows. Repeats get a
 *  "~n" suffix; server list order is stable, so keys are stable too. */
export function duckRowKeys(ducks: { id: string }[]): string[] {
  const seen = new Map<string, number>();
  return ducks.map((d) => {
    const n = seen.get(d.id) ?? 0;
    seen.set(d.id, n + 1);
    return n === 0 ? d.id : `${d.id}~${n + 1}`;
  });
}

/** WebSocket with auto-reconnect; latest frame lands in a mutable ref. */
export class LabClient {
  frame: Frame | null = null;
  connected = false;
  /** Date.now() of the last frame received (or of the open, before the first
   *  one). An OPEN socket is not the same as ARRIVING frames: the lab's duck
   *  loop has died with its socket still open, leaving a green "live" badge
   *  over an empty scene for as long as anyone cared to watch. Consumers age
   *  this out to say "stalled" instead. 0 = not connected. */
  lastFrameAt = 0;
  private ws: WebSocket | null = null;
  private closed = false;
  // Events live in exactly one 25 Hz frame each, so pollers reading `frame`
  // would miss them — accumulate here and let a consumer drain them.
  private pendingEvents: string[] = [];

  constructor(private onStatus?: (connected: boolean) => void) {
    this.connect();
  }

  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  private connect() {
    if (this.closed) return;
    const ws = new WebSocket(LAB_WS);
    this.ws = ws;
    // Every handler ignores sockets that are no longer this.ws: across lab
    // restarts, out-of-order close events used to schedule overlapping
    // connect()s, leaving TWO live sockets — frames arrived on one while
    // sendCmd wrote into the other (a corpse), so buttons/keys silently did
    // nothing even though the page looked live. One socket, one truth.
    ws.onopen = () => {
      if (this.ws !== ws) {
        ws.close();
        return;
      }
      this.connected = true;
      this.lastFrameAt = Date.now();  // start the stall clock at the open, so
      this.onStatus?.(true);          // a socket that never sends is caught
    };
    ws.onmessage = (ev) => {
      if (this.ws !== ws) return;
      const frame: Frame = JSON.parse(ev.data);
      this.frame = frame;
      this.lastFrameAt = Date.now();
      if (frame.events?.length) {
        this.pendingEvents.push(...frame.events);
        if (this.pendingEvents.length > 50)
          this.pendingEvents = this.pendingEvents.slice(-50);
      }
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.connected = false;
      this.lastFrameAt = 0;
      this.onStatus?.(false);
      if (!this.closed && this.reconnectTimer === null) {
        this.reconnectTimer = setTimeout(() => {
          this.reconnectTimer = null;
          this.connect();
        }, 1500);
      }
    };
    ws.onerror = () => ws.close();
  }

  sendCmd(cmd: [number, number, number]) {
    if (this.ws?.readyState === WebSocket.OPEN)
      this.ws.send(JSON.stringify({ cmd }));
  }
  sendReset() {
    if (this.ws?.readyState === WebSocket.OPEN)
      this.ws.send(JSON.stringify({ reset: true }));
  }
  /** Hot-swap `duckId`'s brain to `policyId` (from fetchPolicies).
   *  `showcase` (the chain-level "whole trick" chip) asks the server to
   *  rebuild the duck's env with the behavior's final-stage spawn knobs so
   *  it rehearses the whole trick arc — a no-op for policies without a
   *  curriculum behind them. */
  sendAssign(duckId: string, policyId: string, showcase = false) {
    if (this.ws?.readyState === WebSocket.OPEN)
      this.ws.send(
        JSON.stringify({
          assign: { duck: duckId, policy: policyId, ...(showcase && { showcase: true }) },
        })
      );
  }
  /** Add a helper duck (+2 training envs; warm-restarts the trainer).
   *  Server refusals (no training / cap reached / no snapshot yet / restart
   *  in flight) come back as one-shot event toasts. */
  sendSpawnHelper() {
    if (this.ws?.readyState === WebSocket.OPEN)
      this.ws.send(JSON.stringify({ spawn_helper: true }));
  }
  /** Remove any duck by id. The server guards the edge cases (trainee while
   *  training, helpers during a trainer restart) and refuses via event toasts. */
  sendRemoveDuck(duckId: string) {
    if (this.ws?.readyState === WebSocket.OPEN)
      this.ws.send(JSON.stringify({ remove_duck: { duck: duckId } }));
  }
  /** Spawn a fresh duck running `policyId` (from fetchPolicies). Server caps
   *  the roster at 20; refusals come back as one-shot event toasts.
   *  `showcase` works exactly as in sendAssign. */
  sendSpawnDuck(policyId: string, showcase = false) {
    if (this.ws?.readyState === WebSocket.OPEN)
      this.ws.send(
        JSON.stringify({
          spawn_duck: { policy: policyId, ...(showcase && { showcase: true }) },
        })
      );
  }
  /** Put a BODY on the stage with no policy — it idles (an arm holds HOME
   *  under its zero action, a level-0 body holds its keyframe). The only way
   *  on for a body that ships nothing to assign: MARS before anyone has
   *  trained it, and every Menagerie model. Server refusals (roster full, no
   *  such body) come back as one-shot event toasts. */
  sendSpawnRobot(robot: string) {
    if (this.ws?.readyState === WebSocket.OPEN)
      this.ws.send(JSON.stringify({ spawn_robot: { robot } }));
  }
  /** Drain event lines accumulated since the last call (oldest first). */
  takeEvents(): string[] {
    if (!this.pendingEvents.length) return [];
    const out = this.pendingEvents;
    this.pendingEvents = [];
    return out;
  }
  close() {
    this.closed = true;
    this.ws?.close();
  }
}


// --- practice budget ---------------------------------------------------------
// The teach panel lets the user pick how long a trick practices for. The
// server owns the real arithmetic (viz_server.split_step_budget); these
// mirror it so the panel can PREVIEW a budget — the stage list on screen has
// to be the one that will actually run. Keep the two in step.

/** Bounds the server clamps a chosen budget to (MIN/MAX_STEP_BUDGET). */
export const MIN_STEP_BUDGET = 100_000;
export const MAX_STEP_BUDGET = 40_000_000;

export const clampStepBudget = (steps: number) =>
  Math.min(MAX_STEP_BUDGET, Math.max(MIN_STEP_BUDGET, Math.round(steps)));

/** Scale a curriculum's declared per-stage budgets to a chosen TOTAL, keeping
 *  the stages' ratios. Largest-remainder rounding, so the parts sum to
 *  exactly `total`; every stage keeps at least one step. */
export function splitStepBudget(declared: number[], total: number): number[] {
  const n = declared.length;
  if (n === 0) return [];
  const want = Math.max(Math.round(total), n);
  const base = declared.reduce((s, d) => s + Math.max(0, d), 0);
  const exact = declared.map((d) =>
    base > 0 ? (Math.max(0, d) * want) / base : want / n
  );
  const out = exact.map((x) => Math.max(1, Math.floor(x)));
  const short = want - out.reduce((s, v) => s + v, 0);
  const order = [...out.keys()].sort(
    (a, b) =>
      exact[b] - Math.floor(exact[b]) - (exact[a] - Math.floor(exact[a]))
  );
  for (let i = 0; i < short; i++) out[order[i % n]] += 1;
  let over = short;
  while (over < 0) {
    let j = 0;
    for (let i = 1; i < n; i++) if (out[i] > out[j]) j = i;
    if (out[j] <= 1) break;
    out[j] -= 1;
    over += 1;
  }
  return out;
}

/** The per-stage budgets a launch will actually use: the recipe's declared
 *  steps, scaled to `total` when the user chose one, with explicit per-stage
 *  pins laid on top (mirrors TrainingJob._resolve_stage_steps). */
export function resolveStageSteps(
  declared: number[],
  total: number | null,
  pins: Record<string, number> = {}
): number[] {
  const out = total != null ? splitStepBudget(declared, total) : [...declared];
  for (const [k, v] of Object.entries(pins)) {
    const i = Number(k);
    if (Number.isInteger(i) && i >= 1 && i <= out.length && v > 0) out[i - 1] = v;
  }
  return out;
}
