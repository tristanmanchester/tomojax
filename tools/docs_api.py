"""Generate the documentation site's API pages from the source with griffe.

One MDX page per public module: its docstring, a summary table, then each
exported name with its signature and docstring, and a class's public methods
and properties. griffe reads the source statically, so nothing is imported.
Docstrings are NumPy style: prose with reST inline markup (``code``,
:class:`Name`, ``::`` literal blocks, definition lists) and underlined sections
(Parameters, Returns, Raises, Examples and so on). They become Markdown, with
sections as bold labels rather than headings, so the page's contents list holds
only the documented objects, and with roles linked to the page that documents
their target.

Run with griffe available, for example
``uv run --no-project --with griffe python tools/docs_api.py``.
"""
# griffe is not a project dependency, so its objects are untyped here.
# pyright: reportMissingImports=false, reportAny=false, reportExplicitAny=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import textwrap
from typing import Any

import griffe

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/tristanmanchester/tomojax/blob/main"
# (module, page slug); a name is documented on the first page that exports it.
MODULES = [
    ("tomojax", "tomojax"),
    ("tomojax.io", "io"),
    ("tomojax.corrections", "corrections"),
    ("tomojax.recon", "recon"),
    ("tomojax.alignment", "alignment"),
    ("tomojax.geometry", "geometry"),
    ("tomojax.forward", "forward"),
    ("tomojax.datasets", "datasets"),
    ("tomojax.evaluation", "evaluation"),
    ("tomojax.torch", "torch"),
    ("tomojax.recon.api", "recon-api"),
    ("tomojax.io.api", "io-api"),
    ("tomojax.alignment.api", "alignment-api"),
    ("tomojax.geometry.api", "geometry-api"),
    ("tomojax.forward.api", "forward-api"),
    ("tomojax.corrections.api", "corrections-api"),
]
# The workflow names in reading order; the rest of a module follows its __all__.
FIRST = [
    "load",
    "load_frames",
    "reconstruct",
    "align",
    "project",
    "backproject",
    "save",
    "load_reconstruction",
    "load_alignment",
    "load_simulation",
    "Scan",
    "Frames",
    "Reconstruction",
    "Alignment",
    "Simulation",
]
SKIP = {"__version__"}
# NumPy section titles by how their body is laid out.
ENTRIES = {
    "Parameters",
    "Other Parameters",
    "Attributes",
    "Returns",
    "Yields",
    "Receives",
    "Raises",
    "Warns",
}
SECTIONS = ENTRIES | {"Notes", "Warnings", "References", "See Also", "Examples", "Methods"}
UNDERLINE = re.compile(r"-{3,}$")
DOCTEST = re.compile(r"\s*#\s*doctest:.*$")
ROLE = re.compile(r":(\w+):`(~?)([^`]+)`")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")


def slug(text: str) -> str:
    """Return the anchor github-slugger gives a heading."""
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def _escape(text: str) -> str:
    """Escape what MDX would read as JSX or expressions, outside code spans."""
    parts, last = [], 0
    for match in CODE_SPAN.finditer(text):
        parts += [re.sub(r"([{}<])", r"\\\1", text[last : match.start()]), match.group(0)]
        last = match.end()
    return "".join([*parts, re.sub(r"([{}<])", r"\\\1", text[last:])])


class Pages:
    """The public objects, the page and anchor of each, and their rendering."""

    def __init__(self, package: griffe.Module) -> None:
        self.package = package
        self.urls: dict[str, str] = {}
        self.home: dict[str, str] = {}  # canonical path -> public path it is documented as
        for module, page in MODULES:
            self.urls[module] = f"/api/{page}/"
            for name, obj in self.exports(module):
                public = f"{module}.{name}"
                target = obj.path
                if target in self.home:  # documented on an earlier page
                    self.urls[public] = self.urls[target]
                    continue
                self.home[target] = public
                self.urls[public] = self.urls[target] = f"/api/{page}/#{slug(name)}"
                # fields link to their class; methods and properties to their own entry
                fields = [m for m in obj.members if obj.is_class and not m.startswith("_")]
                entries = {m.name for m in self.members(obj)}
                for member in fields:
                    key = f"{name}.{member}" if member in entries else name
                    self.urls[f"{public}.{member}"] = f"/api/{page}/#{slug(key)}"
                    self.urls[f"{target}.{member}"] = self.urls[f"{public}.{member}"]

    def exports(self, module: str) -> list[tuple[str, Any]]:
        """Return a module's exported names and their resolved objects, in reading order."""
        mod = self.package if module == "tomojax" else self.package[module.removeprefix("tomojax.")]
        names = [str(name) for name in mod.exports or [] if str(name) not in SKIP]
        names.sort(key=lambda name: FIRST.index(name) if name in FIRST else len(FIRST))
        # Loading child modules can replace same-named TYPE_CHECKING re-exports
        # in Griffe's member table. Resolve the declared import path first.
        found = [
            self.package.modules_collection[mod.imports[name]]
            if name in mod.imports
            else mod.members[name]
            for name in names
        ]
        return [(n, self._exported(n, m)) for n, m in zip(names, found, strict=True)]

    @staticmethod
    def _exported(name: str, member: Any) -> Any:
        """Return the exported object (resolving submodules named after their own function)."""
        obj = member.final_target if member.is_alias else member
        if obj.is_module and name in obj.members:
            inner = obj.members[name]
            return inner.final_target if inner.is_alias else inner
        return obj

    @staticmethod
    def members(obj: Any) -> list[Any]:
        """Return a class's public methods and properties, in source order."""
        if not obj.is_class:
            return []
        return [
            member
            for name, member in obj.members.items()
            if not name.startswith("_")
            and not member.is_alias
            and (member.is_function or "property" in member.labels)
        ]

    def link(self, role: str, target: str, context: list[str], *, short: bool) -> str:
        """Render a reST role as a link to its target's page, or as code."""
        shown = target.rsplit(".", 1)[-1] if short else target
        for candidate in [target, *(f"{scope}.{target}" for scope in context)]:
            if candidate in self.urls:
                return f"[`{shown}`]({self.urls[candidate]})"
        return f"`{shown}`" if role != "math" else f"`{target}`"

    def inline(self, text: str, context: list[str]) -> str:
        """Convert reST inline markup to Markdown and escape it for MDX."""
        text = ROLE.sub(lambda m: self.link(m[1], m[3], context, short=bool(m[2])), text)
        text = re.sub(r"``(.+?)``", r"`\1`", text, flags=re.DOTALL)
        return _escape(text)

    def docstring(self, text: str, context: list[str]) -> str:
        """Convert a prose docstring to Markdown."""
        lines, out, i, in_list = text.expandtabs().splitlines(), [], 0, False

        def indent(line: str) -> int:
            return len(line) - len(line.lstrip())

        def block(start: int, base: int) -> tuple[str, int]:
            end = start
            while end < len(lines) and (not lines[end].strip() or indent(lines[end]) > base):
                end += 1
            return textwrap.dedent("\n".join(lines[start:end])).strip("\n"), end

        while i < len(lines):
            line, base = lines[i], indent(lines[i])
            stripped = line.strip()
            follows = i + 1 < len(lines) and lines[i + 1].strip() and indent(lines[i + 1]) > base
            if stripped.endswith("::"):
                head = line.rstrip()[:-1] if stripped != "::" else ""
                body, i = block(i + 1, base)
                out += [self.inline(head, context), "", "```python", body, "```", ""]
                continue
            starts = i == 0 or not lines[i - 1].strip() or in_list
            if follows and stripped and starts and stripped[0] not in "-*":
                body, i = block(i + 1, base)
                term = self.inline(stripped, context)
                out.append(f"- {term}: {self.inline(' '.join(body.split()), context)}")
                in_list = True
                continue
            if in_list and stripped:
                out.append("")
            in_list = False
            out.append(self.inline(line, context))
            i += 1
        return "\n".join(out).strip()

    @staticmethod
    def split(text: str) -> tuple[str, list[tuple[str, str]]]:
        """Split a NumPy docstring into its leading prose and its (title, body) sections."""
        lines = text.expandtabs().splitlines()
        starts = [
            i
            for i in range(len(lines) - 1)
            if lines[i].strip() in SECTIONS and UNDERLINE.match(lines[i + 1].strip())
        ]
        prose = "\n".join(lines[: starts[0]] if starts else lines)
        ends = [*starts[1:], len(lines)][: len(starts)]
        sections = [
            (lines[i].strip(), textwrap.dedent("\n".join(lines[i + 2 : end])).strip("\n"))
            for i, end in zip(starts, ends, strict=True)
        ]
        return prose, sections

    def entries(self, title: str, body: str, context: list[str]) -> list[str]:
        """Render a Parameters-like section as a list of name, type and description."""
        out = []
        for entry in re.split(r"\n(?=\S)", body):
            head, _, desc = entry.partition("\n")
            name, colon, kind = head.partition(" : ")
            if not colon and title in {"Returns", "Yields", "Receives", "Raises", "Warns"}:
                name, kind = "", head
            name, kind = name.strip(), kind.strip()
            label = (
                "**" + ", ".join(f"`{n.strip()}`" for n in name.split(",")) + "**" if name else ""
            )
            if kind:
                typed = f"*{self.inline(kind, context)}*"
                label = f"{label} ({typed})" if name else typed
            text = self.docstring(textwrap.dedent(desc).strip("\n"), context) if desc else ""
            first, _, rest = text.partition("\n")
            if first.startswith(("```", "- ")):  # a block cannot share the label's line
                first, rest = "", text
            lines = [f"- {label}{': ' if first and label else ''}{first}"]
            lines += [f"  {line}" if line else "" for line in rest.splitlines()]
            out += lines
        return [*out, ""]

    @staticmethod
    def doctest(line: str) -> str:
        """Return a doctest line as plain code, its expected output as a comment."""
        line = DOCTEST.sub("", line)
        if line.startswith((">>> ", "... ")):
            return line[4:]
        return "" if line == "..." else f"# {line}"

    def examples(self, body: str, context: list[str]) -> list[str]:
        """Render an Examples section: prose as prose and each ``>>>`` block as code."""
        out = []
        for chunk in re.split(r"\n\s*\n", body):
            if chunk.lstrip().startswith(">>>"):
                out += [
                    "```python",
                    *(self.doctest(line) for line in chunk.splitlines()),
                    "```",
                    "",
                ]
            else:
                out += [self.docstring(chunk, context), ""]
        return out

    def see_also(self, body: str, context: list[str]) -> list[str]:
        """Render a See Also section as links, each with its description."""
        out = []
        for entry in re.split(r"\n(?=\S)", body):
            names, _, desc = entry.partition(" : ")
            links = [
                self.link("obj", name.strip(), context, short=False) for name in names.split(",")
            ]
            desc = " ".join(desc.split()) or " ".join(entry.partition("\n")[2].split())
            out.append(
                f"- {', '.join(links)}" + (f": {self.inline(desc, context)}" if desc else "")
            )
        return [*out, ""]

    def describe(self, text: str, context: list[str]) -> str:
        """Convert a NumPy docstring to Markdown, its sections as bold labels."""
        prose, sections = self.split(text)
        out = [self.docstring(prose, context), ""]
        for title, body in sections:
            out += [f"**{title}**", ""]
            if title in ENTRIES:
                out += self.entries(title, body, context)
            elif title == "Examples":
                out += self.examples(body, context)
            elif title == "See Also":
                out += self.see_also(body, context)
            elif title == "References":
                out += [
                    self.docstring(re.sub(r"^\.\. (?=\[)", "- ", body, flags=re.M), context),
                    "",
                ]
            else:
                out += [self.docstring(body, context), ""]
        return "\n".join(out).strip()

    @staticmethod
    def signature(name: str, obj: Any, *, prefix: str = "") -> str:
        """Format a function's or a class constructor's signature, one line if it fits."""
        params, star, slash = [], False, False
        listed = [p for p in obj.parameters if p.name not in {"self", "cls"}]
        for p in listed:
            kind = p.kind.value
            if kind == "positional-only":
                slash = True
            elif slash:
                params.append("/")
                slash = False
            if kind == "keyword-only" and not star:
                params.append("*")
                star = True
            text = {"variadic positional": "*", "variadic keyword": "**"}.get(kind, "") + p.name
            star = star or kind == "variadic positional"
            if p.annotation is not None:
                text += f": {p.annotation}"
            if p.default is not None and not kind.startswith("variadic"):
                default = str(p.default)
                default = default if len(default) <= 60 else default[:57] + "..."
                text += f" = {default}" if p.annotation is not None else f"={default}"
            params.append(text)
        if slash:
            params.append("/")
        returns = f" -> {obj.returns}" if obj.is_function and obj.returns is not None else ""
        head = f"{prefix}{name}("
        one_line = head + ", ".join(params) + ")" + returns
        if len(one_line) <= 88:
            return one_line
        return head + "\n" + "".join(f"    {p},\n" for p in params) + ")" + returns

    def source(self, obj: Any) -> str:
        """Link to an object's definition on GitHub."""
        path = Path(obj.filepath).resolve().relative_to(ROOT).as_posix()
        return f"[{path}:{obj.lineno}]({GITHUB}/{path}#L{obj.lineno})"

    def summary(self, obj: Any, context: list[str]) -> str:
        """Return the first sentence block of an object's docstring, as Markdown."""
        if not obj.docstring:
            return ""
        first = obj.docstring.value.strip().split("\n\n", 1)[0]
        return " ".join(self.inline(first, context).split())

    def render(self, name: str, obj: Any, module: str) -> list[str]:
        """Render one exported object as Markdown."""
        context = [module, f"{module}.{name}", "tomojax"]
        out = [f"## `{name}`", ""]
        doc = self.describe(obj.docstring.value, context) if obj.docstring else ""
        if obj.is_function or obj.is_class:
            prefix = "class " if obj.is_class else ""
            out += ["```python", self.signature(name, obj, prefix=prefix), "```", ""]
        elif obj.is_attribute:
            value = f" = {obj.value}" if obj.value is not None else ""
            out += ["```python", f"{name}{value}", "```", ""]
        out += [f"<small>Defined in {self.source(obj)}</small>", "", doc, ""]
        for member in self.members(obj):
            heading = f"{name}.{member.name}"
            out += [f"### `{heading}`", ""]
            if "property" in member.labels:
                kind = f": {member.annotation}" if member.annotation is not None else ""
                out += ["```python", f"@property\n{member.name}{kind}", "```", ""]
            else:
                labels = "".join(
                    f"@{label}\n"
                    for label in ("classmethod", "staticmethod")
                    if label in member.labels
                )
                out += ["```python", labels + self.signature(member.name, member), "```", ""]
            if member.docstring:
                out += [self.describe(member.docstring.value, context), ""]
        return out

    def page(self, module: str) -> str:
        """Render a module's page."""
        mod = self.package if module == "tomojax" else self.package[module.removeprefix("tomojax.")]
        context = [module, "tomojax"]
        text = mod.docstring.value.strip() if mod.docstring else ""
        first, _, rest = text.partition("\n\n")
        body = [self.describe(rest, context), "", "| Name | Summary |", "| --- | --- |"]
        rendered = []
        for name, obj in self.exports(module):
            public = f"{module}.{name}"
            body.append(f"| [`{name}`]({self.urls[public]}) | {self.summary(obj, context)} |")
            if self.home[obj.path] == public:
                rendered += self.render(name, obj, module)
        front = {
            "title": module,
            "description": " ".join(ROLE.sub(r"\3", first).replace("``", "").split()),
            "source": Path(mod.filepath).resolve().relative_to(ROOT).as_posix(),
        }
        header = "\n".join(f"{key}: {json.dumps(value)}" for key, value in front.items())
        return f"---\n{header}\n---\n\n" + "\n".join([*body, "", *rendered]).strip() + "\n"


def main() -> None:
    """Write one API page per public module into the site's content."""
    loader = griffe.GriffeLoader(search_paths=[ROOT / "src"])
    package = loader.load("tomojax")
    loader.resolve_aliases(implicit=False, external=None)
    pages = Pages(package)

    out = ROOT / "site" / "content" / "docs" / "api"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for module, page in MODULES:
        _ = (out / f"{page}.mdx").write_text(pages.page(module), encoding="utf-8")
    meta = {"title": "API reference", "pages": [page for _, page in MODULES]}
    _ = (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(MODULES)} API pages to {out}")


if __name__ == "__main__":
    main()
