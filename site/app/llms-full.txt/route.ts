import { getCanonicalUrl, rewriteRelativeMarkdownUrls } from "@/lib/markdown";
import { siteUrl, tagline } from "@/lib/shared";
import { source } from "@/lib/source";

export const revalidate = false;

export async function GET() {
  const sections = await Promise.all(
    source.getPages().map(async (page) => {
      const canonical = getCanonicalUrl(page.url, siteUrl);
      const body = rewriteRelativeMarkdownUrls(
        (await page.data.getText("processed")).trim(),
        siteUrl,
      );
      return `# ${page.data.title}\n\nURL: ${canonical}\n\n${body}`;
    }),
  );

  const output = [
    "# TomoJAX — Full Documentation",
    "",
    `> ${tagline}`,
    "",
    `> For the concise documentation index, see [llms.txt](${siteUrl}/llms.txt).`,
    "",
    sections.join("\n\n---\n\n"),
    "",
  ].join("\n");

  return new Response(output, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
}
