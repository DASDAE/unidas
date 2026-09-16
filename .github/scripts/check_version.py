"""
Ensure a pull request raises unidas' version above the base branch.

The version is a single hardcoded string in src/unidas/__init__.py which
setuptools reads at build time (see pyproject.toml); before 0.3 it lived in
src/unidas.py, which the base branch may still hold. Nothing bumps it automatically, so
without this check two different states of the code can share one version
string, and a released version no longer identifies what was released.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

# The package spelling first; the single-module spelling for an older base.
VERSION_FILES = ("src/unidas/__init__.py", "src/unidas.py")
VERSION_FILE = VERSION_FILES[0]
VERSION_REGEX = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.MULTILINE)


def parse_version(source_code: str, origin: str) -> Version:
    """Pull __version__ out of the module source, or explain why we can't."""
    match = VERSION_REGEX.search(source_code)
    if match is None:
        sys.exit(f"Could not find __version__ in {VERSION_FILE} from {origin}.")
    try:
        return Version(match.group(1))
    except InvalidVersion:
        sys.exit(f"{origin} has an unparsable __version__: {match.group(1)!r}")


def read_base_version(base_ref: str) -> Version:
    """Read the version file as it exists on the base branch."""
    for version_file in VERSION_FILES:
        result = subprocess.run(
            ["git", "show", f"{base_ref}:{version_file}"],
            capture_output=True,
            text=True,
        )
        if not result.returncode:
            return parse_version(result.stdout, base_ref)
    sys.exit(f"Could not read a version file from {base_ref}: {result.stderr.strip()}")


def main() -> None:
    """Compare this branch's version to the base branch's."""
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <base_ref>")
    base_ref = sys.argv[1]
    base_version = read_base_version(base_ref)
    head_file = next((x for x in VERSION_FILES if Path(x).exists()), VERSION_FILE)
    head_version = parse_version(Path(head_file).read_text(), "this branch")
    if head_version <= base_version:
        sys.exit(
            f"Version must increase: {base_ref} is at {base_version}, this branch "
            f"is at {head_version}. Bump __version__ in {VERSION_FILE} so this "
            f"change gets its own version."
        )
    sys.stdout.write(f"Version raised from {base_version} to {head_version}.\n")


if __name__ == "__main__":
    main()
