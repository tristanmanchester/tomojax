import { DocsLayout } from "fumadocs-ui/layouts/docs";
import { ContentFirstContainer } from "@/components/content-first-container";
import { baseOptions } from "@/lib/layout.shared";
import { source } from "@/lib/source";

export default function Layout({ children }: LayoutProps<"/">) {
  return (
    <DocsLayout
      tree={source.getPageTree()}
      slots={{ container: ContentFirstContainer }}
      {...baseOptions()}
    >
      {children}
    </DocsLayout>
  );
}
