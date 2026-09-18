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
 *  "duck" and "g1" are the two hand-built looks. Everything else is
 *  "generic": each geom drawn in its own streamed `rgba`, which is how MARS
 *  arrives Innate orange and a Menagerie model arrives in its MJCF's own
 *  colours with no component of its own. The scene dump does not carry the
 *  look, so this is derived from the id the same way the server derives it.
 */
export function robotLook(id: string): "duck" | "g1" | "generic" {
  if (!id || id === "microduck") return "duck";
  if (id === "g1") return "g1";
  return "generic";
}

/** "teach the duck a trick" vs "teach MARS a task" — Innate's vocabulary for
 *  Innate's robot (`docs/mars-roadmap.md` §4, "vocabulary drift": in their
 *  stack a skill is a task an arm performs, and calling it a trick promises
 *  the wrong thing). A legged body does tricks; anything else does tasks. */
export function trickNoun(kind?: string): "trick" | "task" {
  return !kind || kind === "legged" ? "trick" : "task";
}
