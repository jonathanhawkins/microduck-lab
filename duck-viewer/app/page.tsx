"use client";

import dynamic from "next/dynamic";

// Canvas/WebSocket are browser-only.
const Viewer = dynamic(() => import("@/components/Viewer"), { ssr: false });

export default function Home() {
  return <Viewer />;
}
