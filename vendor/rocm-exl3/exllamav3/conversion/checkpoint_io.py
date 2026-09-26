"""Bounded retry for checkpoint directory renames.

Sibling checkpoint directories only (ckpt, ckpt_old, ckpt_new). No
overwrite, no delete/copy/merge. Retries only PermissionError while the
source directory remains and the destination is absent. Stdlib only.
"""

import os
import time

RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def _validate_checkpoint_paths(source, destination):
    if os.path.islink(source):
        raise NotADirectoryError(f"source is a symlink: {source}")
    if not os.path.isdir(source):
        if not os.path.lexists(source):
            raise FileNotFoundError(f"source not found: {source}")
        raise NotADirectoryError(f"source is not a directory: {source}")
    if os.path.lexists(destination):
        raise FileExistsError(f"destination exists: {destination}")
    src_parent = os.path.realpath(os.path.dirname(os.path.abspath(source)))
    dst_parent = os.path.realpath(os.path.dirname(os.path.abspath(destination)))
    if src_parent != dst_parent:
        raise ValueError(f"different parent directories: {source} -> {destination}")


def rename_checkpoint_dir(source, destination):
    """Rename sibling checkpoint dir; retry only PermissionError."""
    failures = 0
    while True:
        _validate_checkpoint_paths(source, destination)
        try:
            os.rename(source, destination)
        except PermissionError as exc:
            if os.path.lexists(destination) or os.path.islink(source) or not os.path.isdir(source):
                raise
            if failures >= len(RETRY_DELAYS):
                raise
            delay = RETRY_DELAYS[failures]
            print(
                f" -- Checkpoint rename failed (attempt {failures + 1}, PermissionError: {exc}); "
                f"retrying in {delay}s: {source} -> {destination}",
                flush=True,
            )
            time.sleep(delay)
            failures += 1
            continue
        if failures:
            print(
                f" -- Checkpoint rename succeeded after {failures} retries: {source} -> {destination}",
                flush=True,
            )
        return
