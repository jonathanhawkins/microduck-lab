"use client";

import dynamic from "next/dynamic";

// Polls the lab over HTTP — browser-only, same as the other pages.
const TrainPanel = dynamic(() => import("@/components/TrainPanel"), { ssr: false });

export default function TrainPage() {
  return <TrainPanel />;
}
