"""Package the compiled ``tomojax-studio`` binary into a Linux x86_64 wheel and tarball."""

from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path
import re
import subprocess
import tarfile
import tomllib
from typing import cast
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
LICENSE_FILE = ROOT / "LICENSE"
STUDIO_README = ROOT / "studio" / "README.md"

ALLOWED_SYSTEM_LIBS = frozenset(
    {
        "linux-vdso.so.1",
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libm.so.6",
        "libdl.so.2",
        "libpthread.so.0",
        "librt.so.1",
        "libgcc_s.so.1",
    }
)
GLIBC_SYMBOL = re.compile(r"GLIBC_(\d+)\.(\d+)")
MANYLINUX_TAG = re.compile(r"manylinux_(\d+)_(\d+)_")


def project_version() -> str:
    """Read the release version from ``pyproject.toml``."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = cast("dict[str, object]", data["project"])
    return str(project["version"])


def verify_static_linux_binary(binary: Path, platform_tag: str) -> None:
    """Ensure the binary links only standard glibc/libgcc libraries within the tag's floor."""
    ldd = subprocess.run(
        ["ldd", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    unexpected: list[str] = []
    for line in ldd.splitlines():
        name = line.strip().split()[0] if line.strip() else ""
        soname = Path(name).name
        if soname and soname not in ALLOWED_SYSTEM_LIBS:
            unexpected.append(soname)
    if unexpected:
        raise ValueError(
            f"{binary} dynamically links non-system libraries ({', '.join(sorted(unexpected))}); "
            "build Studio with --features static-hdf5"
        )
    if match := MANYLINUX_TAG.match(platform_tag):
        limit = (int(match[1]), int(match[2]))
        version_info = subprocess.run(
            ["readelf", "--version-info", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        versions = [(int(m[1]), int(m[2])) for m in GLIBC_SYMBOL.finditer(version_info)]
        highest = max(versions, default=(0, 0))
        if highest > limit:
            raise ValueError(
                f"{binary} requires GLIBC_{highest[0]}.{highest[1]}, "
                f"exceeding {platform_tag} (GLIBC_{limit[0]}.{limit[1]})"
            )


def _metadata(version: str) -> str:
    readme = STUDIO_README.read_text(encoding="utf-8")
    lines = [
        "Metadata-Version: 2.4",
        "Name: tomojax-studio",
        f"Version: {version}",
        "Summary: Native desktop workspace for TomoJAX tomography projects",
        "Home-page: https://tomojax.dev/studio/",
        "Author-email: Tristan Manchester <tmanchester96@gmail.com>",
        "License-Expression: GPL-3.0-only",
        "License-File: LICENSE",
        "Project-URL: Documentation, https://tomojax.dev/studio/",
        "Project-URL: Repository, https://github.com/tristanmanchester/tomojax",
        "Project-URL: Issues, https://github.com/tristanmanchester/tomojax/issues",
        "Classifier: Development Status :: 4 - Beta",
        "Classifier: Intended Audience :: Science/Research",
        "Classifier: Operating System :: POSIX :: Linux",
        "Classifier: Topic :: Scientific/Engineering :: Physics",
        "Requires-Python: >=3.12",
        "Description-Content-Type: text/markdown",
        "",
        readme,
    ]
    return "\n".join(lines)


def _wheel_file(platform_tag: str) -> str:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: tomojax-build-studio-wheel\n"
        "Root-Is-Purelib: false\n"
        f"Tag: py3-none-{platform_tag}\n"
    )


def _record_line(name: str, payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
    return f"{name},sha256={digest},{len(payload)}"


def build_wheel(binary: Path, version: str, platform_tag: str, out_dir: Path) -> Path:
    """Write a PEP 427 wheel placing ``tomojax-studio`` into the environment's scripts bin."""
    dist_info = f"tomojax_studio-{version}.dist-info"
    data_scripts = f"tomojax_studio-{version}.data/scripts/tomojax-studio"
    wheel_path = out_dir / f"tomojax_studio-{version}-py3-none-{platform_tag}.whl"
    out_dir.mkdir(parents=True, exist_ok=True)

    binary_bytes = binary.read_bytes()
    license_bytes = LICENSE_FILE.read_bytes()
    metadata_bytes = _metadata(version).encode("utf-8")
    wheel_bytes = _wheel_file(platform_tag).encode("utf-8")

    entries: list[tuple[str, bytes, int]] = [
        (data_scripts, binary_bytes, 0o100755),
        (f"{dist_info}/METADATA", metadata_bytes, 0o100644),
        (f"{dist_info}/WHEEL", wheel_bytes, 0o100644),
        (f"{dist_info}/licenses/LICENSE", license_bytes, 0o100644),
    ]
    record_name = f"{dist_info}/RECORD"
    record_lines = [_record_line(name, payload) for name, payload, _ in entries]
    record_lines.append(f"{record_name},,")
    record_bytes = ("\n".join(record_lines) + "\n").encode("utf-8")

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload, mode in (*entries, (record_name, record_bytes, 0o100644)):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = mode << 16
            archive.writestr(info, payload)
    return wheel_path


def build_tarball(binary: Path, version: str, out_dir: Path) -> tuple[Path, Path]:
    """Write a standalone Linux x86_64 tarball and its SHA-256 checksum file."""
    tar_path = out_dir / f"tomojax-studio-{version}-linux-x86_64.tar.gz"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as archive:
        bin_info = archive.gettarinfo(str(binary), arcname="tomojax-studio")
        bin_info.mode = 0o755
        bin_info.uid = bin_info.gid = 0
        bin_info.uname = bin_info.gname = ""
        with binary.open("rb") as handle:
            archive.addfile(bin_info, handle)
        lic_info = archive.gettarinfo(str(LICENSE_FILE), arcname="LICENSE")
        lic_info.mode = 0o644
        lic_info.uid = lic_info.gid = 0
        lic_info.uname = lic_info.gname = ""
        with LICENSE_FILE.open("rb") as handle:
            archive.addfile(lic_info, handle)
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    sha_path = out_dir / f"{tar_path.name}.sha256"
    _ = sha_path.write_text(f"{digest}  {tar_path.name}\n", encoding="utf-8")
    return tar_path, sha_path


def main(argv: list[str] | None = None) -> int:
    """Build the ``tomojax-studio`` wheel and standalone release archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "studio" / "target" / "release" / "tomojax-studio",
        help="Path to the compiled tomojax-studio executable",
    )
    _ = parser.add_argument(
        "--version",
        default=None,
        help="Package version (defaults to project.version in pyproject.toml)",
    )
    _ = parser.add_argument(
        "--platform-tag",
        default="manylinux_2_28_x86_64",
        help="PEP 425 wheel platform tag (default: manylinux_2_28_x86_64)",
    )
    _ = parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "dist-studio",
        help="Output directory for the wheel and tarball",
    )
    _ = parser.add_argument(
        "--check-static",
        action="store_true",
        help="Verify the binary links only standard glibc/libgcc libraries within platform-tag",
    )
    args = parser.parse_args(argv)
    binary = cast("Path", args.binary)
    if not binary.is_file():
        raise FileNotFoundError(f"Studio binary not found: {binary}")
    version = cast("str | None", args.version) or project_version()
    platform_tag = cast("str", args.platform_tag)
    out_dir = cast("Path", args.out_dir)
    if cast("bool", args.check_static):
        verify_static_linux_binary(binary, platform_tag)
    wheel = build_wheel(binary, version, platform_tag, out_dir)
    tarball, sha = build_tarball(binary, version, out_dir)
    print(f"Wrote {wheel}")
    print(f"Wrote {tarball} ({sha.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
