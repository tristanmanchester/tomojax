import { createGetUrl } from "fumadocs-core/source";

export const appName = "TomoJAX";
export const tagline =
  "Parallel-beam, laminography and cone-beam CT reconstruction and alignment in JAX.";
export const docsRoute = "/";
export const docsContentRoute = "/llms.mdx/docs";

export const gitConfig = {
  user: "tristanmanchester",
  repo: "tomojax",
  branch: "main",
};

/** The repository file a page was made from, on GitHub. */
export function getSourceUrl(source: string) {
  return `https://github.com/${gitConfig.user}/${gitConfig.repo}/blob/${gitConfig.branch}/${source}`;
}

const getContentUrl = createGetUrl(docsContentRoute);

export function getPageMarkdownUrl(page: { slugs: string[]; locale?: string }) {
  const segments = [...page.slugs, "content.md"];

  return { segments, url: getContentUrl(segments, page.locale) };
}
