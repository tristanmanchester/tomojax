import type { MetadataRoute } from "next";
import { siteUrl } from "@/lib/shared";
import { source } from "@/lib/source";

export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  return source.getPages().map((page) => ({
    url: new URL(page.url.endsWith("/") ? page.url : `${page.url}/`, siteUrl).toString(),
    changeFrequency: "weekly",
    priority: page.slugs.length === 0 ? 1 : 0.7,
  }));
}
