const SITE_BASE = "https://tomojax.dev";

/** Return the `.md` twin path for a page's URL slugs (e.g. `/` -> `/index.md`, `/quickstart/` -> `/quickstart.md`). */
export function getPageMarkdownTwinPath(slugs: string[]): string {
  return slugs.length === 0 ? "/index.md" : `/${slugs.join("/")}.md`;
}

/** Resolve a site path against `baseUrl` with a canonical trailing slash. */
export function getCanonicalUrl(
  pageUrl: string,
  baseUrl: string = SITE_BASE,
): string {
  const normalized = pageUrl.endsWith("/") ? pageUrl : `${pageUrl}/`;
  return new URL(normalized, baseUrl).toString();
}

/** Rewrite root-relative Markdown links and images (`](/...)`) outside code blocks/spans to absolute URLs. */
export function rewriteRelativeMarkdownUrls(
  markdown: string,
  baseUrl: string = SITE_BASE,
): string {
  const origin = baseUrl.replace(/\/+$/, "");
  const lines = markdown.split("\n");
  const out: string[] = [];
  let fenceChar = "";
  let fenceLen = 0;

  for (const line of lines) {
    const trimmed = line.trimStart();
    if (fenceChar) {
      const closeMatch = trimmed.match(/^(`{3,}|~{3,})\s*$/);
      if (
        closeMatch &&
        closeMatch[1][0] === fenceChar &&
        closeMatch[1].length >= fenceLen
      ) {
        fenceChar = "";
        fenceLen = 0;
      }
      out.push(line);
      continue;
    }

    const openMatch = trimmed.match(/^(`{3,}|~{3,})/);
    if (openMatch) {
      fenceChar = openMatch[1][0];
      fenceLen = openMatch[1].length;
      out.push(line);
      continue;
    }

    out.push(rewriteLineOutsideInlineCode(line, origin));
  }

  return out.join("\n");
}

function rewriteLineOutsideInlineCode(line: string, origin: string): string {
  let result = "";
  let i = 0;
  let plainStart = 0;

  while (i < line.length) {
    if (line[i] === "`") {
      let ticks = 0;
      while (i + ticks < line.length && line[i + ticks] === "`") {
        ticks += 1;
      }
      const marker = "`".repeat(ticks);
      const closeIdx = line.indexOf(marker, i + ticks);
      if (closeIdx !== -1) {
        result += rewritePlainSegment(line.slice(plainStart, i), origin);
        result += line.slice(i, closeIdx + ticks);
        i = closeIdx + ticks;
        plainStart = i;
        continue;
      }
      i += ticks;
      continue;
    }
    i += 1;
  }

  result += rewritePlainSegment(line.slice(plainStart), origin);
  return result;
}

function rewritePlainSegment(segment: string, origin: string): string {
  // Rewrite Markdown link/image destinations `](/path)` (skip protocol-relative `](//...)`).
  let rewritten = segment.replace(
    /(\]\()(\/(?!\/)[^\s)]+)/g,
    (_match, prefix: string, path: string) => `${prefix}${origin}${path}`,
  );
  // Rewrite root-relative HTML href/src attributes (e.g. `<a href="/...">`).
  rewritten = rewritten.replace(
    /(\b(?:href|src)=["'])(\/(?!\/)[^"']+)(["'])/g,
    (_match, prefix: string, path: string, quote: string) =>
      `${prefix}${origin}${path}${quote}`,
  );
  return rewritten;
}

export interface MarkdownDocumentInput {
  title: string;
  description?: string;
  canonicalUrl: string;
  lastUpdated: string;
  body: string;
  baseUrl?: string;
}

/** Render a standalone Markdown document with YAML frontmatter, llms.txt directive, and absolute links. */
export function formatMarkdownDocument({
  title,
  description,
  canonicalUrl,
  lastUpdated,
  body,
  baseUrl = SITE_BASE,
}: MarkdownDocumentInput): string {
  const origin = baseUrl.replace(/\/+$/, "");
  const frontmatterLines = ["---", `title: ${JSON.stringify(title)}`];
  if (description) {
    frontmatterLines.push(`description: ${JSON.stringify(description)}`);
  }
  frontmatterLines.push(
    `canonical: ${JSON.stringify(canonicalUrl)}`,
    `last-updated: ${JSON.stringify(lastUpdated)}`,
    "---",
  );

  const absoluteBody = rewriteRelativeMarkdownUrls(body.trim(), origin);
  return [
    frontmatterLines.join("\n"),
    "",
    `# ${title}`,
    "",
    `> For the complete documentation index, see [llms.txt](${origin}/llms.txt).`,
    "",
    absoluteBody,
    "",
  ].join("\n");
}
