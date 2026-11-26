import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
} from "fumadocs-ui/layouts/docs/page";
import { createRelativeLink } from "fumadocs-ui/mdx";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getMDXComponents } from "@/components/mdx";
import {
  AskAIPopover,
  CopyMarkdownButton,
  GitHubSourceLink,
} from "@/components/page-actions";
import { getPageMarkdownTwinPath } from "@/lib/markdown";
import {
  appName,
  getPageMarkdownUrl,
  getSourceUrl,
  siteUrl,
} from "@/lib/shared";
import { source } from "@/lib/source";

export default async function Page(props: PageProps<"/[[...slug]]">) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const MDX = page.data.body;
  const markdownUrl = getPageMarkdownUrl(page).url;
  const markdownTwinUrl = `${siteUrl}${getPageMarkdownTwinPath(page.slugs)}`;
  // API pages list every method; their table of contents keeps to the exported names.
  const toc =
    page.slugs[0] === "api"
      ? page.data.toc.filter((item) => item.depth <= 2)
      : page.data.toc;

  return (
    <DocsPage toc={toc} full={page.data.full}>
      <div className="sr-only">
        For AI agents: see the complete TomoJAX documentation index at{" "}
        <a href={`${siteUrl}/llms.txt`}>llms.txt</a>. This page is also
        available as Markdown at <a href={markdownTwinUrl}>{markdownTwinUrl}</a>{" "}
        or by sending <code>Accept: text/markdown</code>.
      </div>
      <DocsTitle>{page.data.title}</DocsTitle>
      <DocsDescription className="mb-0">
        {page.data.description}
      </DocsDescription>
      <div className="flex flex-row flex-wrap gap-2 items-center border-b pb-6">
        <CopyMarkdownButton markdownPath={markdownUrl} />
        <AskAIPopover
          markdownUrl={`${siteUrl}${markdownUrl}`}
          title={page.data.title}
        />
        {page.data.source && (
          <GitHubSourceLink href={getSourceUrl(page.data.source)} />
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

  const markdownTwinUrl = `${siteUrl}${getPageMarkdownTwinPath(page.slugs)}`;

  return {
    title: page.slugs.length === 0 ? { absolute: appName } : page.data.title,
    description: page.data.description,
    alternates: {
      canonical: page.url,
      types: {
        "text/markdown": markdownTwinUrl,
      },
    },
  };
}
