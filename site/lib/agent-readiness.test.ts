import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { describe, it } from "node:test";
import { onRequest } from "../functions/_middleware.js";
import {
  formatMarkdownDocument,
  getCanonicalUrl,
  getPageMarkdownTwinPath,
  rewriteRelativeMarkdownUrls,
} from "./markdown.ts";

const OUT_DIR = path.resolve(import.meta.dirname, "..", "out");

function walkFiles(
  dir: string,
  predicate: (file: string) => boolean,
): string[] {
  const results: string[] = [];
  if (!fs.existsSync(dir)) return results;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...walkFiles(full, predicate));
    } else if (predicate(full)) {
      results.push(full);
    }
  }
  return results;
}

function hasRootRelativeMarkdownLinks(markdown: string): boolean {
  const lines = markdown.split("\n");
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
      continue;
    }
    const openMatch = trimmed.match(/^(`{3,}|~{3,})/);
    if (openMatch) {
      fenceChar = openMatch[1][0];
      fenceLen = openMatch[1].length;
      continue;
    }
    // Strip inline code spans before testing.
    const withoutCode = line.replace(/(`+)[\s\S]*?\1/g, "");
    if (/\]\(\/(?!\/)/.test(withoutCode)) {
      return true;
    }
  }
  return false;
}

function measureContentStartRatio(html: string): number {
  const bodyMatch = html.match(/<body[^>]*>([\s\S]*)<\/body>/i);
  const body = bodyMatch ? bodyMatch[1] : html;
  const cleaned = body
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, "")
    .replace(/<template\b[^>]*>[\s\S]*?<\/template>/gi, "");

  const h1Match = cleaned.match(/<h1\b[^>]*>/i);
  assert.ok(
    h1Match && h1Match.index !== undefined,
    "Page must contain an <h1>",
  );

  const stripTags = (fragment: string) =>
    fragment
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim();

  const beforeText = stripTags(cleaned.slice(0, h1Match.index));
  const totalText = stripTags(cleaned);
  assert.ok(totalText.length > 0, "Page body must contain visible text");
  return beforeText.length / totalText.length;
}

describe("markdown helpers", () => {
  it("computes .md twin paths and canonical URLs", () => {
    assert.equal(getPageMarkdownTwinPath([]), "/index.md");
    assert.equal(getPageMarkdownTwinPath(["quickstart"]), "/quickstart.md");
    assert.equal(
      getPageMarkdownTwinPath(["api", "tomojax"]),
      "/api/tomojax.md",
    );
    assert.equal(getCanonicalUrl("/"), "https://tomojax.dev/");
    assert.equal(
      getCanonicalUrl("/quickstart"),
      "https://tomojax.dev/quickstart/",
    );
  });

  it("rewrites root-relative links outside code while preserving code blocks and spans", () => {
    const input = [
      "See [Quickstart](/quickstart/) and ![watch](/images/watch.png).",
      "Keep external [GitHub](https://github.com/tristanmanchester/tomojax) and `#anchor`.",
      "Inline code `[keep](/local)` stays literal, then [API](/api/tomojax/#load) rewrites.",
      "```markdown",
      "[inside-fence](/should-not-change/)",
      "```",
      "After fence: [Lab CT](/lab-ct/).",
    ].join("\n");

    const output = rewriteRelativeMarkdownUrls(input, "https://tomojax.dev");
    assert.match(
      output,
      /\[Quickstart\]\(https:\/\/tomojax\.dev\/quickstart\/\)/,
    );
    assert.match(
      output,
      /!\[watch\]\(https:\/\/tomojax\.dev\/images\/watch\.png\)/,
    );
    assert.match(output, /`\[keep\]\(\/local\)`/);
    assert.match(
      output,
      /\[API\]\(https:\/\/tomojax\.dev\/api\/tomojax\/#load\)/,
    );
    assert.match(output, /\[inside-fence\]\(\/should-not-change\/\)/);
    assert.match(output, /\[Lab CT\]\(https:\/\/tomojax\.dev\/lab-ct\/\)/);
  });

  it("formats documents with YAML frontmatter, llms.txt directive, and absolute links", () => {
    const doc = formatMarkdownDocument({
      title: "Quickstart",
      description: "From a scan file to inspected slices.",
      canonicalUrl: "https://tomojax.dev/quickstart/",
      lastUpdated: "2026-11-26",
      body: "Read [Installation](/installation/).",
    });

    assert.ok(doc.startsWith("---\n"));
    assert.match(doc, /^title: "Quickstart"$/m);
    assert.match(
      doc,
      /^description: "From a scan file to inspected slices\."$/m,
    );
    assert.match(doc, /^canonical: "https:\/\/tomojax\.dev\/quickstart\/"$/m);
    assert.match(doc, /^last-updated: "2026-11-26"$/m);
    assert.match(
      doc,
      /^> For the complete documentation index, see \[llms\.txt\]\(https:\/\/tomojax\.dev\/llms\.txt\)\.$/m,
    );
    assert.match(
      doc,
      /\[Installation\]\(https:\/\/tomojax\.dev\/installation\/\)/,
    );
  });
});

describe("Cloudflare Pages middleware", () => {
  const assets = new Map<string, string>([
    [
      "/llms.mdx/docs/content.md",
      '---\ntitle: "TomoJAX"\n---\n\n# TomoJAX\n\nHome body.\n',
    ],
    [
      "/llms.mdx/docs/quickstart/content.md",
      '---\ntitle: "Quickstart"\n---\n\n# Quickstart\n\nGuide body.\n',
    ],
    [
      "/llms.mdx/docs/api/tomojax/content.md",
      '---\ntitle: "tomojax"\n---\n\n# tomojax\n\nAPI body.\n',
    ],
  ]);

  function makeContext(
    url: string,
    headers: Record<string, string> = {},
    nextResponse: Response = new Response(
      "<html><body>Not found</body></html>",
      {
        status: 404,
        headers: { "content-type": "text/html; charset=utf-8" },
      },
    ),
  ) {
    return {
      request: new Request(url, { headers }),
      env: {
        ASSETS: {
          fetch: async (target: URL | string) => {
            const pathname = new URL(target).pathname;
            const body = assets.get(pathname);
            if (body === undefined) {
              return new Response("Not Found", { status: 404 });
            }
            return new Response(body, {
              status: 200,
              headers: { "content-type": "text/markdown; charset=utf-8" },
            });
          },
        },
      },
      next: async () => nextResponse,
    };
  }

  it("redirects pages.dev and www to tomojax.dev", async () => {
    const res = await onRequest(
      makeContext("https://www.tomojax.dev/quickstart/"),
    );
    assert.equal(res.status, 301);
    assert.equal(
      res.headers.get("location"),
      "https://tomojax.dev/quickstart/",
    );
  });

  it("serves /index.md and /<slug>.md twins with canonical Link header", async () => {
    const home = await onRequest(makeContext("https://tomojax.dev/index.md"));
    assert.equal(home.status, 200);
    assert.equal(
      home.headers.get("content-type"),
      "text/markdown; charset=utf-8",
    );
    assert.equal(
      home.headers.get("link"),
      '<https://tomojax.dev/>; rel="canonical"',
    );
    assert.match(await home.text(), /# TomoJAX/);

    const api = await onRequest(
      makeContext("https://tomojax.dev/api/tomojax.md"),
    );
    assert.equal(api.status, 200);
    assert.equal(
      api.headers.get("link"),
      '<https://tomojax.dev/api/tomojax/>; rel="canonical"',
    );
    assert.match(await api.text(), /# tomojax/);
  });

  it("negotiates Markdown on Accept: text/markdown", async () => {
    const res = await onRequest(
      makeContext("https://tomojax.dev/quickstart/", {
        accept: "text/markdown, text/html;q=0.8",
      }),
    );
    assert.equal(res.status, 200);
    assert.equal(
      res.headers.get("content-type"),
      "text/markdown; charset=utf-8",
    );
    assert.match(await res.text(), /# Quickstart/);
  });

  it("returns HTTP 404 with Markdown error body for missing paths when agents request Markdown", async () => {
    const probe = await onRequest(
      makeContext("https://tomojax.dev/__ora-404-probe-j2pp23eg", {
        accept: "text/markdown",
      }),
    );
    assert.equal(probe.status, 404);
    assert.equal(
      probe.headers.get("content-type"),
      "text/markdown; charset=utf-8",
    );
    const probeBody = await probe.text();
    assert.ok(probeBody.length >= 20);
    assert.match(probeBody, /https:\/\/tomojax\.dev\/llms\.txt/);

    const missingMd = await onRequest(
      makeContext("https://tomojax.dev/does-not-exist.md"),
    );
    assert.equal(missingMd.status, 404);
    assert.equal(
      missingMd.headers.get("content-type"),
      "text/markdown; charset=utf-8",
    );
    assert.match(await missingMd.text(), /https:\/\/tomojax\.dev\/llms\.txt/);
  });

  it("preserves redirects from context.next() even when Accept: text/markdown is set", async () => {
    const redirect = await onRequest(
      makeContext(
        "https://tomojax.dev/alignment-guide/",
        { accept: "text/markdown" },
        Response.redirect("https://tomojax.dev/align-a-scan/", 301),
      ),
    );
    assert.equal(redirect.status, 301);
    assert.equal(
      redirect.headers.get("location"),
      "https://tomojax.dev/align-a-scan/",
    );
  });

  it("appends Link rel=alternate type=text/markdown on HTML doc responses", async () => {
    const html = await onRequest(
      makeContext(
        "https://tomojax.dev/quickstart/",
        { accept: "text/html" },
        new Response("<!doctype html><html><body>Hi</body></html>", {
          status: 200,
          headers: { "content-type": "text/html; charset=utf-8" },
        }),
      ),
    );
    assert.equal(html.status, 200);
    assert.match(
      html.headers.get("link") || "",
      /<https:\/\/tomojax\.dev\/quickstart\.md>; rel="alternate"; type="text\/markdown"/,
    );
  });
});

describe("built site output (site/out)", {
  skip: !fs.existsSync(OUT_DIR),
}, () => {
  it("emits valid root and section llms.txt files", () => {
    const root = fs.readFileSync(path.join(OUT_DIR, "llms.txt"), "utf8");
    assert.ok(root.startsWith("# TomoJAX\n"), "llms.txt must start with H1");
    assert.match(root, /^> .+/m, "llms.txt must have a blockquote summary");
    assert.match(root, /^## When to use TomoJAX$/m);
    assert.match(root, /^## Get started$/m);
    assert.match(root, /^## API reference$/m);
    assert.match(root, /https:\/\/tomojax\.dev\/guides\/llms\.txt/);
    assert.match(root, /https:\/\/tomojax\.dev\/api\/llms\.txt/);
    assert.equal(hasRootRelativeMarkdownLinks(root), false);

    for (const rel of ["guides/llms.txt", "api/llms.txt"]) {
      const content = fs.readFileSync(path.join(OUT_DIR, rel), "utf8");
      assert.match(content, /^# /);
      assert.match(content, /^> .+/m);
      assert.match(content, /^## /m);
      assert.equal(hasRootRelativeMarkdownLinks(content), false);
    }
  });

  it("emits frontmatter, llms.txt directive, and portable links in all generated Markdown pages", () => {
    const mdFiles = walkFiles(path.join(OUT_DIR, "llms.mdx", "docs"), (file) =>
      file.endsWith("content.md"),
    );
    assert.ok(
      mdFiles.length >= 40,
      `Expected >=40 markdown files, got ${mdFiles.length}`,
    );

    for (const file of mdFiles) {
      const text = fs.readFileSync(file, "utf8");
      assert.ok(text.startsWith("---\n"), `${file} missing YAML frontmatter`);
      assert.match(text, /^title: /m, `${file} missing title`);
      assert.match(
        text,
        /^canonical: "https:\/\/tomojax\.dev\//m,
        `${file} missing canonical`,
      );
      assert.match(
        text,
        /^last-updated: "\d{4}-\d{2}-\d{2}"$/m,
        `${file} missing last-updated`,
      );
      assert.match(
        text,
        /^> For the complete documentation index, see \[llms\.txt\]\(https:\/\/tomojax\.dev\/llms\.txt\)\.$/m,
        `${file} missing llms.txt directive blockquote`,
      );
      assert.equal(
        hasRootRelativeMarkdownLinks(text),
        false,
        `${file} still contains root-relative markdown links`,
      );
    }
  });

  it("emits alternate markdown links, llms.txt HTML directive, and <10% content start position", () => {
    const htmlFiles = walkFiles(OUT_DIR, (file) => {
      const rel = path.relative(OUT_DIR, file);
      return (
        file.endsWith("index.html") &&
        !rel.startsWith("404") &&
        !rel.startsWith("_not-found")
      );
    });
    assert.ok(
      htmlFiles.length >= 40,
      `Expected >=40 HTML pages, got ${htmlFiles.length}`,
    );

    let worstRatio = 0;
    let worstPage = "";
    for (const file of htmlFiles) {
      const html = fs.readFileSync(file, "utf8");
      const rel = path.relative(OUT_DIR, file);
      assert.match(
        html,
        /<link[^>]+rel="alternate"[^>]+type="text\/markdown"[^>]+href="https:\/\/tomojax\.dev\/[^"]+\.md"/,
        `${rel} missing <link rel="alternate" type="text/markdown">`,
      );
      assert.match(
        html,
        /https:\/\/tomojax\.dev\/llms\.txt/,
        `${rel} missing llms.txt directive`,
      );
      const ratio = measureContentStartRatio(html);
      if (ratio > worstRatio) {
        worstRatio = ratio;
        worstPage = rel;
      }
      assert.ok(
        ratio < 0.1,
        `${rel} content starts too late (${(ratio * 100).toFixed(1)}% >= 10%)`,
      );
    }
    assert.ok(worstPage.length > 0 && worstRatio < 0.1);
  });
});
