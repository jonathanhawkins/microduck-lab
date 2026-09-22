// How a robot id becomes something a person can see: an emoji, and the
// material set the stage draws it with.
//
// `docs/mars-roadmap.md` §6.4: "one `lib/robots.ts` map (id -> emoji, look,
// chip label) and a per-geom material `kind` sent by the server, so
// `Duck.tsx` maps kind -> material and no robot needs its own component".
// The chip LABEL already has one home (`lib/activeRobot.robotChipLabel`, put
// there when the three panels were unified), so it stays there and this file
// owns the emoji and nothing else — two definitions of one label is the bug
// that file was written to fix.

/** The emoji a robot's chip wears.
 *
 *  By ID first, because the three built-in bodies have identities worth
 *  showing (🦆 duck, 🤖 G1, 🛸 MARS) — and by KIND after, because a body this
 *  build has never heard of arrives from a plugin or a Menagerie download and
 *  still has to render as something. The fallback is 🤖: a robot nobody has
 *  drawn an emoji for is still a robot, and a blank chip reads as a broken
 *  one.
 *
 *  `kind` comes from the lab (`LabRobot.kind` — "legged" | "wheeled" |
 *  "generic"), so a new BUILT-IN body can also just pick up its kind's emoji
 *  without an entry here. */
export function robotEmoji(id: string, kind?: string): string {
  const byId: Record<string, string> = {
    microduck: "🦆",
    g1: "🤖",
    mars: "🛸",
  };
  if (byId[id]) return byId[id];
  const byKind: Record<string, string> = {
    wheeled: "🛸",
    legged: "🤖",
  };
  return (kind && byKind[kind]) || "🤖";
}

/** Which material table the stage paints a body with — the server's own
 *  answer (`Body.look()`), carried here so the viewer has one name for it.
 *
 *  "duck", "g1" and "mars" are the three hand-built looks. Everything else is
 *  "generic": each geom drawn in its own streamed `rgba`, which is how a
 *  Menagerie model arrives in its MJCF's own colours with no component of its
 *  own. The scene dump does not carry the look, so this is derived from the
 *  id the same way the server derives it.
 *
 *  MARS was "generic" until its own table existed, and looked it: the
 *  server's charcoal-0.16 chassis and black head are within a few percent of
 *  the lab stage's #101216 backdrop, so the robot read as a floating orange
 *  arm (components/MarsLook.tsx has the whole argument). A body only earns a
 *  table when a generic paint fails ON A STAGE — that is the bar, not
 *  "shipped with the lab". */
export function robotLook(id: string): "duck" | "g1" | "generic" | "mars" {
  if (!id || id === "microduck") return "duck";
  if (id === "g1") return "g1";
  if (id === "mars") return "mars";
  return "generic";
}

/** "teach the duck a trick" vs "teach MARS a task" — Innate's vocabulary for
 *  Innate's robot (`docs/mars-roadmap.md` §4, "vocabulary drift": in their
 *  stack a skill is a task an arm performs, and calling it a trick promises
 *  the wrong thing). A legged body does tricks; anything else does tasks. */
export function trickNoun(kind?: string): "trick" | "task" {
  return !kind || kind === "legged" ? "trick" : "task";
}

/** "2 ducks", "1 MARS", "2 G1" — a count of one body kind, for a menu.
 *
 *  The plural rule is the noun's own CASE, and it is a rule rather than a
 *  table because the nouns come from the lab (`robot_noun`, which a plugin
 *  body also answers). A lowercase noun is a common one and takes an s; a
 *  noun with a capital in it is a name or an acronym and does not — nobody
 *  writes "MARSs" or "G1s". A body whose noun is empty falls back to its id,
 *  which the server already does before it gets here. */
export function robotCount(n: number, noun: string): string {
  const common = noun === noun.toLowerCase();
  return `${n} ${noun}${n === 1 || !common ? "" : "s"}`;
}

/** Is this body the DUCK? The one question the viewer asks often enough to
 *  be worth a name: only a duck quacks (lib/quack.ts), and a scenario's own
 *  entries leave `robot` off when they mean the duck (world/scenario.Duck
 *  defaults it to "microduck"). */
export function isDuck(robot?: string | null): boolean {
  return !robot || robot === "microduck";
}
