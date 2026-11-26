import { createFromSource } from "fumadocs-core/search/server";
import { normalizeSpellingToken } from "@/lib/search-spelling";
import { source } from "@/lib/source";

export const revalidate = false;

export const { staticGET: GET } = createFromSource(source, {
  tokenizer: {
    language: "english",
    stemmer: normalizeSpellingToken,
  },
});
