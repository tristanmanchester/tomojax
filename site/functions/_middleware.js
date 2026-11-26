const SITE_URL = "https://tomojax.dev";

function markdownHeaders(body, canonicalPath) {
  return {
    "content-type": "text/markdown; charset=utf-8",
    "x-markdown-tokens": String(Math.ceil(body.length / 4)),
    link: `<${SITE_URL}${canonicalPath}>; rel="canonical"`,
    vary: "Accept",
    "cache-control": "public, max-age=300",
  };
}

function markdownNotFound(pathname) {
  const body = [
    "# 404 — Page Not Found",
    "",
    `The requested path \`${pathname}\` does not exist on TomoJAX documentation.`,
    "",
    "Use these entry points to locate documentation pages:",
    "",
    `- [Documentation index (llms.txt)](${SITE_URL}/llms.txt)`,
    `- [Homepage (Markdown)](${SITE_URL}/index.md)`,
    `- [Guides index](${SITE_URL}/guides/llms.txt)`,
    `- [Python API reference index](${SITE_URL}/api/llms.txt)`,
    `- [Sitemap](${SITE_URL}/sitemap.xml)`,
    "",
  ].join("\n");

  return new Response(body, {
    status: 404,
    headers: {
      "content-type": "text/markdown; charset=utf-8",
      "x-markdown-tokens": String(Math.ceil(body.length / 4)),
      vary: "Accept",
      "cache-control": "public, max-age=60",
    },
  });
}

function docMarkdownAssetPath(slugPath) {
  return slugPath
    ? `/llms.mdx/docs${slugPath}/content.md`
    : "/llms.mdx/docs/content.md";
}

function htmlMarkdownTwinPath(pathname) {
  const trimmed = pathname.replace(/\/+$/, "");
  return trimmed === "" ? "/index.md" : `${trimmed}.md`;
}

// Markdown for agents:
// - Explicit `.md` twin URLs (`/index.md`, `/<slug>.md`) return the page's Markdown.
// - Requests with `Accept: text/markdown` receive the page's Markdown (or a 404 Markdown body).
// - HTML doc responses advertise their `.md` twin via a `Link: rel="alternate"` header.
export async function onRequest(context) {
  const url = new URL(context.request.url);
  // The production pages.dev address and www move to the custom domain; previews stay.
  if (
    url.hostname === "tomojax.pages.dev" ||
    url.hostname === "www.tomojax.dev"
  ) {
    url.hostname = "tomojax.dev";
    return Response.redirect(url.toString(), 301);
  }

  const pathname = url.pathname;
  const accept = context.request.headers.get("accept") || "";
  const wantsMarkdown = /\btext\/markdown\b/i.test(accept);

  // 1. Explicit `.md` twin URLs (excluding `/llms...` static routes).
  if (/\.md$/i.test(pathname) && !pathname.startsWith("/llms")) {
    const withoutExt = pathname.slice(0, -3);
    const slugPath = withoutExt === "/index" ? "" : withoutExt;
    const canonicalPath = slugPath ? `${slugPath}/` : "/";
    const target = new URL(docMarkdownAssetPath(slugPath), url);
    const response = await context.env.ASSETS.fetch(target);
    if (!response.ok) return markdownNotFound(pathname);

    const body = await response.text();
    return new Response(body, {
      headers: markdownHeaders(body, canonicalPath),
    });
  }

  // 2. Content negotiation via `Accept: text/markdown`.
  if (wantsMarkdown) {
    const trimmed = pathname.replace(/\/+$/, "");
    const hasExtension = /\.[a-z0-9]+$/i.test(trimmed);
    if (!trimmed.startsWith("/llms") && !hasExtension) {
      const target = new URL(docMarkdownAssetPath(trimmed), url);
      const response = await context.env.ASSETS.fetch(target);
      if (response.ok) {
        const body = await response.text();
        const canonicalPath = trimmed ? `${trimmed}/` : "/";
        return new Response(body, {
          headers: markdownHeaders(body, canonicalPath),
        });
      }
    }

    const nextResponse = await context.next();
    if (nextResponse.status === 404) {
      return markdownNotFound(pathname);
    }
    return nextResponse;
  }

  // 3. Normal responses: attach `rel="alternate" type="text/markdown"` on HTML doc pages.
  const response = await context.next();
  const contentType = response.headers.get("content-type") || "";
  if (
    response.status === 200 &&
    contentType.includes("text/html") &&
    !pathname.startsWith("/llms") &&
    !pathname.startsWith("/_next")
  ) {
    const headers = new Headers(response.headers);
    headers.append(
      "link",
      `<${SITE_URL}${htmlMarkdownTwinPath(pathname)}>; rel="alternate"; type="text/markdown"`,
    );
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  }

  return response;
}
