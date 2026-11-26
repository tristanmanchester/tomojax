"use client";
import { experimental_useSearch } from "fumadocs-core/search/client";
import { staticClient } from "fumadocs-core/search/client/orama-static";
import {
  SearchDialog,
  SearchDialogClose,
  SearchDialogContent,
  SearchDialogHeader,
  SearchDialogIcon,
  SearchDialogInput,
  SearchDialogList,
  SearchDialogOverlay,
  type SharedProps,
} from "fumadocs-ui/components/dialog/search";
import { useI18n } from "fumadocs-ui/contexts/i18n";
import { useCallback, useMemo } from "react";
import {
  expandHighlightQuery,
  normalizeSearchQuery,
} from "@/lib/search-spelling";

export default function DefaultSearchDialog(props: SharedProps) {
  const { locale } = useI18n(); // (optional) for i18n
  const client = useMemo(() => staticClient({ locale }), [locale]);
  const run = useCallback(
    (query: string) => client.search(normalizeSearchQuery(query)),
    [client],
  );
  const search = experimental_useSearch(run);
  const data = useMemo(
    () =>
      search.data
        ? {
            ...search.data,
            query: expandHighlightQuery(search.data.query),
          }
        : search.data,
    [search.data],
  );

  return (
    <SearchDialog {...search} data={data} {...props}>
      <SearchDialogOverlay />
      <SearchDialogContent>
        <SearchDialogHeader>
          <SearchDialogIcon />
          <SearchDialogInput />
          <SearchDialogClose />
        </SearchDialogHeader>
        <SearchDialogList />
      </SearchDialogContent>
    </SearchDialog>
  );
}
