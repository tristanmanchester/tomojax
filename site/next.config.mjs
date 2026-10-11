import { createMDX } from "fumadocs-mdx/next";

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  output: "export",
  // Every page is a folder with an index.html, which any static host serves.
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default withMDX(config);
