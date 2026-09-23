"""Publish v4 generations as immutable directories behind one atomic reference.

The shared `scripts/artifact_publication.py` replaces the managed directories
one at a time, so a reader arriving mid-switch can see `beir/` from the new
build beside `audits/` from the old one. v1-v3 keep that behavior; v4 does
not use it.

Here each build is staged under `generations/.building-<id>/`, validated,
renamed to `generations/<id>/`, validated again under its final identity, and
only then made visible by replacing the `current` symlink in a single
`os.replace`. A reader sees either the old complete generation or the new
complete generation, never a mixture.

    building -> validated -> active -> superseded
             \\-> failed (removed; never active)

Superseded generations are kept: removal waits for a separate cleanup policy.
"""

from __future__ import annotations

import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

from scripts.v4.generation import (
    BUILDING_PREFIX,
    CURRENT,
    GENERATIONS,
    new_generation_id,
    validate_generation,
)

__all__ = [
    "BUILDING_PREFIX",
    "PublicationError",
    "Published",
    "activate_generation",
    "publish",
]

T = TypeVar("T")


class PublicationError(RuntimeError):
    """The dataset root cannot accept a publication or activation."""


@dataclass(frozen=True)
class Published(Generic[T]):
    generation_id: str
    path: Path
    #: The generation this publication superseded, or None for the first.
    previous_generation_id: str | None
    summary: T


def publish(
    output_root: str | Path,
    build: Callable[[Path, str], T],
    *,
    generation_id: str | None = None,
) -> Published[T]:
    """Build, validate, and atomically activate one new generation.

    `build(staging, generation_id)` writes the complete generation into
    `staging` and must record `generation_id` in its manifest. Any failure —
    in the build, in validation, or in the switch — removes the new
    generation and leaves `current` exactly as it was.
    """
    root = Path(output_root)
    identity = generation_id or new_generation_id()
    generations = root / GENERATIONS
    generations.mkdir(parents=True, exist_ok=True)
    previous = _current_generation_id(root)

    final = generations / identity
    staging = generations / f"{BUILDING_PREFIX}{identity}"
    if final.exists() or staging.exists():
        raise PublicationError(f"generation {identity} already exists under {generations}")

    try:
        summary = build(staging, identity)
        validate_generation(staging, check_directory_identity=False)
        os.rename(staging, final)
        validate_generation(final)
        _switch_current(root, identity)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(final, ignore_errors=True)
        raise
    return Published(identity, final, previous, summary)


def activate_generation(output_root: str | Path, generation_id: str) -> Path:
    """Point `current` at an existing generation — the rollback path.

    The generation is revalidated first, so a superseded generation that has
    since been damaged cannot be reactivated.
    """
    root = Path(output_root)
    target = root / GENERATIONS / generation_id
    if generation_id.startswith(".") or os.sep in generation_id or not target.is_dir():
        raise PublicationError(f"no published generation {generation_id!r} under {root}")
    validate_generation(target)
    _current_generation_id(root)  # refuses a non-symlink `current`
    _switch_current(root, generation_id)
    return target


def _current_generation_id(root: Path) -> str | None:
    pointer = root / CURRENT
    if pointer.is_symlink():
        return Path(os.readlink(pointer)).name
    if pointer.exists():
        raise PublicationError(
            f"{pointer} exists but is not a symlink; refusing to replace it. "
            "It may be a dataset from the legacy in-place publisher."
        )
    return None


def _switch_current(root: Path, generation_id: str) -> None:
    """Replace `current` in one atomic rename of a prepared symlink."""
    temporary = root / f".{CURRENT}.{secrets.token_hex(4)}"
    # Relative, so the dataset root can be moved or mounted elsewhere.
    temporary.symlink_to(os.path.join(GENERATIONS, generation_id))
    try:
        os.replace(temporary, root / CURRENT)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

