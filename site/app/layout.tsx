import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Provider } from "@/components/provider";
import { appName, tagline } from "@/lib/shared";
import "./global.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

export const metadata: Metadata = {
  title: { template: `%s | ${appName}`, default: appName },
  description: tagline,
};

export default function Layout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${mono.variable}`}
      suppressHydrationWarning
    >
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
