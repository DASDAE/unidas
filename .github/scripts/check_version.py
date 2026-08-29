"""
Ensure a pull request raises unidas' version above the base branch.

The version is a single hardcoded string in src/unidas.py which setuptools
reads at build time (see pyproject.toml). Nothing bumps it automatically, so
without this check two different states of the code can share one version
string, and a released version no longer identifies what was released.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

VERSION_FILE = "src/unidas.py"
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
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{VERSION_FILE}"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        sys.exit(
            f"Could not read {VERSION_FILE} from {base_ref}: {result.stderr.strip()}"
        )
    return parse_version(result.stdout, base_ref)


def main() -> None:
    """Compare this branch's version to the base branch's."""
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <base_ref>")
    base_ref = sys.argv[1]
    base_version = read_base_version(base_ref)
    head_version = parse_version(Path(VERSION_FILE).read_text(), "this branch")
    if head_version <= base_version:
        sys.exit(
            f"Version must increase: {base_ref} is at {base_version}, this branch "
            f"is at {head_version}. Bump __version__ in {VERSION_FILE} so this "
            f"change gets its own version."
        )
    sys.stdout.write(f"Version raised from {base_version} to {head_version}.\n")


if __name__ == "__main__":
    main()
