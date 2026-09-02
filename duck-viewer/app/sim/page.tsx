"use client";

import dynamic from "next/dynamic";

// Canvas/WebSocket are browser-only.
const SimViewer = dynamic(() => import("@/components/SimViewer"), { ssr: false });

export default function SimPage() {
  return <SimViewer />;
}
