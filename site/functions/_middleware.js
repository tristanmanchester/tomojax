// Markdown for agents: a request that prefers text/markdown gets the page's
// Markdown (the same text as /llms.mdx/docs/<page>/content.md) instead of HTML.
export async function onRequest(context) {
  const url = new URL(context.request.url);
  // The production pages.dev address and www move to the custom domain; previews stay.
  if (url.hostname === "tomojax.pages.dev" || url.hostname === "www.tomojax.dev") {
    url.hostname = "tomojax.dev";
    return Response.redirect(url.toString(), 301);
  }
  const accept = context.request.headers.get("accept") || "";
  if (!/\btext\/markdown\b/.test(accept)) return context.next();

  const path = url.pathname.replace(/\/+$/, "");
  if (path.startsWith("/llms") || /\.[a-z0-9]+$/i.test(path)) return context.next();
  const target = new URL(`/llms.mdx/docs${path}/content.md`, url);
  const response = await context.env.ASSETS.fetch(target);
  if (!response.ok) return context.next();

  const body = await response.text();
  return new Response(body, {
    headers: {
      "content-type": "text/markdown; charset=utf-8",
      "x-markdown-tokens": String(Math.ceil(body.length / 4)),
      vary: "Accept",
      "cache-control": "public, max-age=300",
    },
  });
}
