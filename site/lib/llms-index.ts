import type * as PageTree from "fumadocs-core/page-tree";
import { getCanonicalUrl } from "@/lib/markdown";
import { gitConfig, siteUrl, tagline } from "@/lib/shared";
import { source } from "@/lib/source";

interface PageEntry {
  title: string;
  url: string;
  description: string;
}

interface DocSection {
  title: string;
  pages: PageEntry[];
  isApi: boolean;
}

function nodeLabel(name: unknown): string {
  return typeof name === "string" ? name : "";
}

function collectSections(): DocSection[] {
  const tree = source.getPageTree();
  const sections: DocSection[] = [];
  let current: DocSection | undefined;

  const startSection = (title: string, isApi: boolean): DocSection => {
    const next: DocSection = { title, pages: [], isApi };
    sections.push(next);
    return next;
  };

  const addPageNode = (node: PageTree.Item, target: DocSection) => {
    const page = source.getNodePage(node);
    const title = page?.data.title ?? nodeLabel(node.name);
    const description = (page?.data.description ?? "").trim();
    const url = getCanonicalUrl(node.url, siteUrl);
    target.pages.push({ title, url, description });
  };

  const walkFolder = (folder: PageTree.Folder, target: DocSection) => {
    if (folder.index) {
      addPageNode(folder.index, target);
    }
    for (const child of folder.children) {
      if (child.type === "page") {
        addPageNode(child, target);
      } else if (child.type === "folder") {
        walkFolder(child, target);
      }
    }
  };

  for (const node of tree.children) {
    if (node.type === "separator") {
      const label = nodeLabel(node.name) || "Documentation";
      current = startSection(label, label.toLowerCase().includes("api"));
    } else if (node.type === "page") {
      if (!current) current = startSection("Documentation", false);
      addPageNode(node, current);
    } else if (node.type === "folder") {
      const label = nodeLabel(node.name) || "Documentation";
      const isApi =
        label.toLowerCase().includes("api") || (current?.isApi ?? false);
      if (!current) current = startSection(label, isApi);
      walkFolder(node, current);
    }
  }

  return sections.filter((section) => section.pages.length > 0);
}

function renderSections(sections: DocSection[]): string[] {
  const out: string[] = [];
  for (const section of sections) {
    out.push(`## ${section.title}`, "");
    for (const page of section.pages) {
      const suffix = page.description ? `: ${page.description}` : "";
      out.push(`- [${page.title}](${page.url})${suffix}`);
    }
    out.push("");
  }
  return out;
}

const repoUrl = `https://github.com/${gitConfig.user}/${gitConfig.repo}`;

/** Build the root `/llms.txt` index following the llms.txt specification. */
export function buildRootLlmsTxt(): string {
  const sections = collectSections();
  const lines = [
    "# TomoJAX",
    "",
    `> ${tagline} Exact-transpose CUDA forward and adjoint projectors, analytical and iterative solvers (Fourier, FBP/FDK, CGLS, SIRT/OS-SART, MLEM/OSEM, FISTA-TV, SPDHG-TV), joint volume-and-pose alignment, CLI, local stdio MCP server, and native Studio desktop workspace.`,
    "",
    `Every documentation page is also available as Markdown by appending \`.md\` to its URL (or [\`/index.md\`](${siteUrl}/index.md) for the homepage), or by sending \`Accept: text/markdown\`.`,
    "",
    "## When to use TomoJAX",
    "",
    "Reach for TomoJAX when a user needs local, GPU-accelerated (or CPU) X-ray tomography reconstruction, alignment, or differentiable projection on Linux:",
    "",
    "- **Reconstruct real or synthetic scans** in parallel-beam, laminography, or lab cone-beam CT (including multi-orbit `ConeSegments`) using Fourier-slice inverse, FBP/FDK, CGLS, SIRT/OS-SART, MLEM/OSEM, FISTA-TV, or SPDHG-TV.",
    "- **Load and calibrate facility or vendor scans** directly from NXtomo (ESRF, Diamond), APS Data Exchange (APS, PSI TOMCAT, Elettra SYRMEP), Nikon `.xtekct`, Zeiss `.txrm`, ESRF Bliss/EDF/PyHST, SkyScan, PSI cSAXS, or TIFF stacks.",
    "- **Fix misalignment and motion artifacts** by jointly solving for the volume and rotation center, setup tilt/roll, per-view translations, or per-view rotations with `tj.align` or `tomojax align`.",
    "- **Differentiate through tomography projectors** with exact-transpose CUDA kernels in JAX (`jax.grad`, `jvp`, `vjp`) or PyTorch (`tomojax.torch.Projector`).",
    "- **Reconstruct volumes larger than GPU memory** using automatic host-slab streaming and multi-GPU execution.",
    "",
    "How an agent should invoke TomoJAX:",
    "",
    "- **Interactive agent workflows on local scan data**: connect to the local stdio MCP server (`tomojax mcp` from `tomojax[mcp,cuda12]`). Scans and volumes stay resident on the GPU between tool calls, and `view` returns labeled slice PNGs for visual inspection.",
    "- **Scripted pipelines and batch reconstruction**: run the `tomojax` CLI (`tomojax inspect`, `correct`, `align`, `reconstruct`, `export`, `simulate`).",
    "- **Custom solvers, training loops, or multi-GPU scripts**: call the Python API (`import tomojax as tj`).",
    "- **Not a hosted cloud API**: TomoJAX runs locally beside the scan files and GPU, with no cloud backend or API keys.",
    "",
    "## Section indexes",
    "",
    `- [Guides index](${siteUrl}/guides/llms.txt): Task-oriented guides for installation, reconstruction, correction, alignment, interfaces, and benchmarks.`,
    `- [API reference index](${siteUrl}/api/llms.txt): Module-by-module Python API reference for \`tomojax\` and its public submodules.`,
    `- [Full documentation](${siteUrl}/llms-full.txt): All TomoJAX documentation concatenated into one Markdown document.`,
    "",
    ...renderSections(sections),
    "## Packages and source",
    "",
    `- [GitHub repository](${repoUrl}): Source code, issue tracker, benchmarks, examples, and Agent Skill (\`plugins/tomojax/\`).`,
    "- [PyPI package (`tomojax`)](https://pypi.org/project/tomojax/): Python library, CLI (`tomojax`), and MCP server (`tomojax[mcp]`).",
    "- [PyPI package (`tomojax-studio`)](https://pypi.org/project/tomojax-studio/): Prebuilt Linux desktop workspace (`tomojax studio`).",
    "",
  ];
  return lines.join("\n");
}

/** Build the `/guides/llms.txt` section index covering task and package guides. */
export function buildGuidesLlmsTxt(): string {
  const sections = collectSections().filter((section) => !section.isApi);
  const lines = [
    "# TomoJAX Guides",
    "",
    "> Task-oriented guides for installing TomoJAX, loading beamline and lab CT scans, correcting artifacts, aligning geometry, running the CLI, MCP server, and Studio desktop workspace, and reviewing benchmarks.",
    "",
    `For the full site index including when-to-use guidance and the Python API reference, see [llms.txt](${siteUrl}/llms.txt).`,
    "",
    ...renderSections(sections),
  ];
  return lines.join("\n");
}

/** Build the `/api/llms.txt` section index covering the Python API reference. */
export function buildApiLlmsTxt(): string {
  const sections = collectSections().filter((section) => section.isApi);
  const lines = [
    "# TomoJAX Python API Reference",
    "",
    "> Module-by-module Python API reference for `tomojax` and its public submodules (`io`, `corrections`, `recon`, `alignment`, `geometry`, `forward`, `datasets`, `evaluation`, `progress`, `interactive`, `torch`).",
    "",
    `For the full site index including task guides and when-to-use guidance, see [llms.txt](${siteUrl}/llms.txt).`,
    "",
    ...renderSections(sections),
  ];
  return lines.join("\n");
}
