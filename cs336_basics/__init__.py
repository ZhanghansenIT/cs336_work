import importlib.metadata
import re
from pathlib import Path

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    # Fallback: read version from pyproject.toml if package is not installed
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if match:
            __version__ = match.group(1)
        else:
            __version__ = "unknown"
    else:
        __version__ = "unknown"
