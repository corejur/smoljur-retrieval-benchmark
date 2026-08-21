"""Publish a complete dataset generation with rollback on failure."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

Result = TypeVar("Result")

MANAGED_DIRECTORIES = (
    "queries",
    "documents",
    "data_preparation_output",
    "chunks",
)


def publish_generation(
    output_directory: str | Path,
    build: Callable[[Path], Result],
    *,
    managed_directories: Sequence[str] = MANAGED_DIRECTORIES,
) -> Result:
    """Build off-path, then publish managed directories with rollback."""
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError(f"refusing to publish through symlink: {output}")

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.generation.", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        result = build(staging)
        output.mkdir(parents=True, exist_ok=True)
        backup = staging / ".previous"
        backup.mkdir()
        replaced: list[str] = []
        installed: list[str] = []
        try:
            for name in managed_directories:
                source = staging / name
                if not source.is_dir():
                    raise ValueError(
                        f"dataset generation omitted managed directory {name!r}"
                    )
                destination = output / name
                if destination.is_symlink():
                    raise ValueError(
                        f"refusing to replace symlinked artifact directory: "
                        f"{destination}"
                    )
                if destination.exists():
                    os.replace(destination, backup / name)
                    replaced.append(name)
                os.replace(source, destination)
                installed.append(name)
        except BaseException:
            for name in reversed(installed):
                destination = output / name
                if destination.exists():
                    os.replace(destination, staging / name)
            for name in reversed(replaced):
                os.replace(backup / name, output / name)
            raise
        return result
