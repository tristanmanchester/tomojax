import { llms, loader } from "fumadocs-core/source";
import { metaSchema, pageSchema } from "fumadocs-core/source/schema";
import { defineDocs } from "fumadocs-mdx/macro";
import { z } from "zod";
import { docsRoute } from "./shared";

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
  renderPage: async (page) => `# ${page.data.title} (${page.url})

${await page.data.getText("processed")}`,
});
