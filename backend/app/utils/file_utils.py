import os


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_safe_relative_path(base_dir: str, target_path: str) -> bool:
    """Guard against path traversal when serving files back to clients."""
    base = os.path.realpath(base_dir)
    target = os.path.realpath(target_path)
    return os.path.commonpath([base]) == os.path.commonpath([base, target])
