import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "P&ID KG App",
  description: "Next.js app for P&ID extraction, querying, discrepancy checks, and graph exploration",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
