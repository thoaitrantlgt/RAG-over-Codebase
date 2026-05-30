import subprocess
from pathlib import Path


def changed_files(
    *,
    repo_path: str | Path,
    base: str,
    head: str = "HEAD",
) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "diff", "--name-only", base, head],
        check=True,
        capture_output=True,
        text=True,
    )
    return normalize_changed_files(result.stdout.splitlines())


def normalize_changed_files(paths: list[str]) -> list[str]:
    normalized = []
    for path in paths:
        clean = path.strip().replace("\\", "/")
        if clean:
            normalized.append(clean)
    return sorted(set(normalized))
