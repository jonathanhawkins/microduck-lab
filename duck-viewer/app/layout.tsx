import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Microduck Lab",
  description: "Live Three.js viewer for Microduck training runs",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#101216" }}>{children}</body>
    </html>
  );
}
