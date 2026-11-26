"use client";
import { Container } from "fumadocs-ui/layouts/docs/slots/container";
import type { ComponentProps } from "react";

/**
 * Renders the page content ahead of the mobile header and desktop sidebar in
 * DOM order. Because every Fumadocs DocsLayout slot binds to an explicit CSS
 * named grid area (`sidebar`, `header`, `main`, `toc`), visual presentation is
 * unchanged while HTML-to-Markdown extractors reach the main article first.
 */
export function ContentFirstContainer({
  children,
  ...props
}: ComponentProps<typeof Container>) {
  // Fumadocs LayoutBody passes: [header, sidebar, tabs, pageChildren, aiChat].
  const reordered =
    Array.isArray(children) && children.length >= 4
      ? [
          children[3],
          children[0],
          children[1],
          children[2],
          ...children.slice(4),
        ]
      : children;

  return <Container {...props}>{reordered}</Container>;
}
