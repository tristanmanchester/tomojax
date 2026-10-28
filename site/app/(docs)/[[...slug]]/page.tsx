import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
  MarkdownCopyButton,
  ViewOptionsPopover,
} from "fumadocs-ui/layouts/docs/page";
import { createRelativeLink } from "fumadocs-ui/mdx";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getMDXComponents } from "@/components/mdx";
import { appName, getPageMarkdownUrl, getSourceUrl } from "@/lib/shared";
import { source } from "@/lib/source";

export default async function Page(props: PageProps<"/[[...slug]]">) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const MDX = page.data.body;
  const markdownUrl = getPageMarkdownUrl(page).url;
  // API pages list every method; their table of contents keeps to the exported names.
  const toc =
    page.slugs[0] === "api"
      ? page.data.toc.filter((item) => item.depth <= 2)
      : page.data.toc;

  return (
    <DocsPage toc={toc} full={page.data.full}>
      <DocsTitle>{page.data.title}</DocsTitle>
      <DocsDescription className="mb-0">
        {page.data.description}
      </DocsDescription>
      <div className="flex flex-row gap-2 items-center border-b pb-6">
        <MarkdownCopyButton markdownUrl={markdownUrl} />
        {page.data.source && (
          <ViewOptionsPopover
            markdownUrl={markdownUrl}
            githubUrl={getSourceUrl(page.data.source)}
          />
        )}
      </div>
      <DocsBody>
        <MDX
          components={getMDXComponents({
            a: createRelativeLink(source, page),
          })}
        />
      </DocsBody>
    </DocsPage>
  );
}

export async function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(
  props: PageProps<"/[[...slug]]">,
): Promise<Metadata> {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  return {
    title: page.slugs.length === 0 ? { absolute: appName } : page.data.title,
    description: page.data.description,
    alternates: { canonical: page.url },
  };
}
