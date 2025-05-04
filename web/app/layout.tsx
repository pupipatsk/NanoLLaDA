import type React from "react";
import "./globals.css";
import type { Metadata } from "next";
import { Outfit, Noto_Sans_Thai } from "next/font/google";
import { ThemeProvider } from "@/components/theme-provider";

// Modern primary font
const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
});

// Font with good Thai language support
const notoSansThai = Noto_Sans_Thai({
  subsets: ["thai"],
  variable: "--font-noto-thai",
  weight: ["300", "400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Thai Text Summarizer",
  description: "A modern tool for summarizing Thai text",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="th"
      suppressHydrationWarning
      className={`${outfit.variable} ${notoSansThai.variable}`}
    >
      <body className="font-sans">
        <ThemeProvider attribute="class" defaultTheme="light">
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
