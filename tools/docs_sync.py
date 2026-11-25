"""Copy the Markdown guides into the documentation site as MDX pages.

The Markdown files in the repository stay the source of truth. This script
writes them to ``site/content/docs`` with front matter, lays out the sidebar
in ``meta.json``, rewrites relative links to site routes (or to GitHub for
files outside the site) and copies the images they show into ``site/public``.
"""

from __future__ import annotations

import json
from pathlib import Path
import posixpath
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
GITHUB = "https://github.com/tristanmanchester/tomojax"
IMAGE_SUFFIXES = (".png", ".gif", ".jpg", ".jpeg", ".svg", ".webp")

# The sidebar, by what a reader wants to do. Each page is
# (source, slug, title or None for the source's heading, description).
SECTIONS: list[tuple[str, list[tuple[str, str, str | None, str]]]] = [
    (
        "Get started",
        [
            (
                "README.md",
                "index",
                "TomoJAX",
                "Parallel-beam, laminography and cone-beam CT reconstruction and alignment in JAX.",
            ),
            ("docs/installation.md", "installation", None, "CPU or CUDA dependencies."),
            ("docs/quickstart.md", "quickstart", None, "From a scan file to inspected slices."),
            (
                "docs/beamline-exports.md",
                "beamline-exports",
                None,
                "First slices from ESRF, TOMCAT/SYRMEP and I13 exports.",
            ),
            (
                "docs/synthetic-tomography.md",
                "synthetic-tomography",
                None,
                "Simulate data and call the Python API.",
            ),
            ("examples/README.md", "examples", "Examples", "The runnable example scripts."),
            ("docs/gallery.md", "gallery", None, "Figures from phantoms and real scans."),
            ("docs/README.md", "guide-map", "Documentation map", "Every guide by task."),
        ],
    ),
    (
        "Reconstruct",
        [
            (
                "docs/real-laminography.md",
                "real-laminography",
                None,
                "Measured geometry for parallel and laminography scans, including TIFF stacks.",
            ),
            (
                "docs/lab-ct.md",
                "lab-ct",
                None,
                "Import, calibrate and reconstruct a lab cone-beam scan.",
            ),
            (
                "studio/README.md",
                "studio",
                "TomoJAX Studio",
                "Native workspace for persistent tomography projects, previews and 3D viewing.",
            ),
            (
                "src/tomojax/cli/README.md",
                "command-line",
                "Command line",
                "The commands, their conventions and --config files.",
            ),
            (
                "docs/pytorch.md",
                "pytorch",
                None,
                "Use the projector as a differentiable PyTorch operator.",
            ),
            (
                "docs/mcp.md",
                "mcp",
                None,
                "Let an AI agent drive TomoJAX through the Model Context Protocol.",
            ),
        ],
    ),
    (
        "Align",
        [
            (
                "docs/align-a-scan.md",
                "align-a-scan",
                None,
                "Choose what to fit: the centre, drift, the setup angles or sample rotation.",
            ),
            (
                "docs/held-out-evaluation.md",
                "held-out-evaluation",
                None,
                "Split views or detector columns to test a fit on data it did not see.",
            ),
        ],
    ),
    (
        "Correct data",
        [
            (
                "src/tomojax/corrections/README.md",
                "corrections",
                "Correction steps",
                "From detector counts to line integrals, and steps on line integrals.",
            ),
        ],
    ),
    (
        "Package guides",
        [
            (
                "src/tomojax/geometry/README.md",
                "geometry",
                "Geometry",
                "Grids, detectors, geometries and poses.",
            ),
            (
                "src/tomojax/forward/README.md",
                "forward-projection",
                "Forward projection",
                "Projector model and derivative choices.",
            ),
            (
                "src/tomojax/recon/README.md",
                "reconstruction",
                "Reconstruction",
                "Algorithms and their configuration.",
            ),
            (
                "src/tomojax/io/README.md",
                "io",
                "IO",
                "Datasets, acquisition metadata and TIFF angle files.",
            ),
            (
                "src/tomojax/alignment/README.md",
                "alignment-configuration",
                "Alignment configuration",
                "The joint solver behind tj.align and its numerical contracts.",
            ),
            (
                "docs/protocol.md",
                "protocol",
                None,
                "JSON Lines worker protocol and standalone progress contract.",
            ),
            (
                "docs/progress-metrics.md",
                "progress-metrics",
                None,
                "Scalar diagnostics emitted during reconstruction and alignment.",
            ),
        ],
    ),
    (
        "Large scans",
        [
            (
                "docs/large-scans.md",
                "large-scans",
                None,
                "Scans larger than the GPU, and several GPUs.",
            ),
        ],
    ),
    (
        "Scope",
        [
            (
                "docs/support-matrix.md",
                "support-matrix",
                None,
                "Geometries, units, algorithms and GPU scope.",
            ),
            (
                "docs/known-limitations.md",
                "known-limitations",
                None,
                "What does not work yet, or needs checking.",
            ),
        ],
    ),
    (
        "Measurements",
        [
            (
                "docs/measurements.md",
                "measurements",
                None,
                "Measured speed, accuracy and memory, against ASTRA and TIGRE.",
            ),
        ],
    ),
    (
        "Contributing",
        [
            (
                "CONTRIBUTING.md",
                "contributing",
                None,
                "Tests, packaging and repository conventions.",
            ),
        ],
    ),
]

REPO_URL = re.compile(rf"{re.escape(GITHUB)}/(?:blob|tree|raw)/main/")
LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")


def _routes() -> dict[str, str]:
    """Map each source file to its site route."""
    return {
        source: "/" if slug == "index" else f"/{slug}/"
        for _, pages in SECTIONS
        for source, slug, _, _ in pages
    }


def _escape(text: str) -> str:
    """Escape the characters MDX would read as JSX, outside code spans."""
    out: list[str] = []
    last = 0
    for match in CODE_SPAN.finditer(text):
        out.append(re.sub(r"([{}<])", r"\\\1", text[last : match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(re.sub(r"([{}<])", r"\\\1", text[last:]))
    return "".join(out)


def _rewrite(line: str, source: str, routes: dict[str, str], images: set[str]) -> str:
    """Point a line's relative links at site routes, site images or GitHub."""

    def replace(match: re.Match[str]) -> str:
        bang, text, target = match[1], match[2], match[3]
        base = posixpath.dirname(source)
        if repo := REPO_URL.match(target):
            # Links written for GitHub (the README is also the PyPI page) are
            # relative to the repository root.
            target, base = target[repo.end() :], ""
        elif re.match(r"[a-z]+:", target) or target.startswith("#"):
            return match.group(0)
        path, _, anchor = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(base, path))
        suffix = f"#{anchor}" if anchor else ""
        if resolved.lower().endswith(IMAGE_SUFFIXES):
            images.add(resolved)
            url = f"/images/{Path(resolved).name}"
        elif resolved in routes:
            url = routes[resolved] + suffix
        else:
            kind = "tree" if (ROOT / resolved).is_dir() else "blob"
            url = f"{GITHUB}/{kind}/main/{resolved}{suffix}"
        return f"{bang}[{text}]({url})"

    return LINK.sub(replace, line)


def convert(
    source: str, title: str | None, description: str, routes: dict[str, str], images: set[str]
) -> str:
    """Return one Markdown file as an MDX page with front matter."""
    lines = (ROOT / source).read_text(encoding="utf-8").splitlines()
    body: list[str] = []
    in_fence, heading = False, None
    for line in lines:
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            body.append(line)
        elif in_fence:
            body.append(line)
        elif heading is None and line.startswith("# "):
            heading = line[2:].strip()
        else:
            body.append(_escape(_rewrite(line, source, routes, images)))
    front = {
        "title": title or heading or Path(source).stem,
        "description": description,
        "source": source,
    }
    header = "\n".join(f"{key}: {json.dumps(value)}" for key, value in front.items())
    return f"---\n{header}\n---\n\n" + "\n".join(body).strip() + "\n"


def main() -> None:
    """Write the guides, the sidebar and the images into the site."""
    content = SITE / "content" / "docs"
    public_images = SITE / "public" / "images"
    for generated in (content, public_images):
        shutil.rmtree(generated, ignore_errors=True)
        generated.mkdir(parents=True)

    routes, images, sidebar = _routes(), set[str](), list[str]()
    for section, pages in SECTIONS:
        sidebar.append(f"---{section}---")
        for source, slug, title, description in pages:
            page = convert(source, title, description, routes, images)
            _ = (content / f"{slug}.mdx").write_text(page, encoding="utf-8")
            sidebar.append(slug)
    sidebar += ["---API reference---", "...api"]
    meta: dict[str, object] = {"title": "TomoJAX", "root": True, "pages": sidebar}
    _ = (content / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    for image in sorted(images):
        _ = shutil.copy2(ROOT / image, public_images / Path(image).name)
    print(f"Wrote {len(routes)} guides and {len(images)} images to {SITE}")


if __name__ == "__main__":
    main()
