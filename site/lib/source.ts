import { llms, loader } from "fumadocs-core/source";
import { metaSchema, pageSchema } from "fumadocs-core/source/schema";
import { defineDocs } from "fumadocs-mdx/macro";
import { z } from "zod";
import { formatMarkdownDocument, getCanonicalUrl } from "./markdown";
import { docsRoute, siteUrl } from "./shared";

const buildDate = new Date().toISOString().slice(0, 10);

// Pages are generated from the repository by tools/docs_sync.py and
// tools/docs_api.py; `source` is the file each one came from.
const docs = defineDocs({
  dir: "content/docs",
  docs: {
    schema: pageSchema.extend({ source: z.string().optional() }),
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
});

export const source = loader({
  baseUrl: docsRoute,
  source: docs.toFumadocsSource(),
  plugins: [],
});

export const docsLlms = llms(source, {
  renderPage: async (page) =>
    formatMarkdownDocument({
      title: page.data.title,
      description: page.data.description,
      canonicalUrl: getCanonicalUrl(page.url, siteUrl),
      lastUpdated: buildDate,
      body: await page.data.getText("processed"),
      baseUrl: siteUrl,
    }),
});
