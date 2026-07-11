"""Load and validate the documentation recovery catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


VALID_DISPOSITIONS = {"rewrite", "current-api", "redirect", "remove"}
VALID_STATES = {"validated", "adapted", "reference"}


class CatalogError(ValueError):
    """Raised when the documentation catalog is incomplete or inconsistent."""


@dataclass(frozen=True)
class Validation:
    state: str
    fork_version: str
    date: str | None = None
    hardware: str | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class Tutorial:
    id: str
    title: str
    category: str
    source: str
    slug: str
    hardware: list[str]
    validation: Validation
    video_id: str | None = None
    smoke_command: list[str] | None = None


@dataclass(frozen=True)
class Route:
    upstream: str
    snapshot: str
    disposition: str
    replacement: str


@dataclass(frozen=True)
class Catalog:
    schema_version: int
    sources: dict[str, Any]
    routes: list[Route]
    tutorials: list[Tutorial]


def _require_unique(values: list[str], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise CatalogError(f"duplicate {label}: {value}")
        seen.add(value)


def load_catalog(path: Path, *, repo_root: Path) -> Catalog:
    """Load ``path`` and validate all source and route invariants."""

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise CatalogError("catalog schema_version must be 1")

    routes = [Route(**item) for item in raw.get("routes", [])]
    tutorials = [
        Tutorial(
            **{key: value for key, value in item.items() if key != "validation"},
            validation=Validation(**item["validation"]),
        )
        for item in raw.get("tutorials", [])
    ]

    _require_unique([route.upstream for route in routes], "route")
    _require_unique([tutorial.id for tutorial in tutorials], "tutorial id")
    _require_unique([tutorial.slug for tutorial in tutorials], "tutorial slug")

    for route in routes:
        if not route.upstream.startswith("/kubetorch/"):
            raise CatalogError(f"route must begin with /kubetorch/: {route.upstream}")
        if route.disposition not in VALID_DISPOSITIONS:
            raise CatalogError(f"invalid route disposition: {route.disposition}")
        if not route.snapshot.isdigit() or len(route.snapshot) != 14:
            raise CatalogError(f"invalid Wayback timestamp: {route.snapshot}")
        if not route.replacement:
            raise CatalogError(f"route has no replacement: {route.upstream}")

    for tutorial in tutorials:
        source = repo_root / tutorial.source
        if not source.is_file():
            raise CatalogError(f"tutorial source does not exist: {tutorial.source}")
        if tutorial.validation.state not in VALID_STATES:
            raise CatalogError(
                f"invalid validation state for {tutorial.id}: {tutorial.validation.state}"
            )
        slug = PurePosixPath(tutorial.slug)
        if (
            slug.is_absolute()
            or not slug.parts
            or slug.parts[0] != "tutorials"
            or ".." in slug.parts
            or "\\" in tutorial.slug
        ):
            raise CatalogError(f"invalid tutorial slug: {tutorial.slug}")
        if tutorial.video_id and any(char.isspace() for char in tutorial.video_id):
            raise CatalogError(f"invalid YouTube id for {tutorial.id}")
        if tutorial.validation.state == "validated":
            required = ("date", "hardware", "evidence")
            missing = [
                field for field in required if not getattr(tutorial.validation, field)
            ]
            if missing:
                raise CatalogError(
                    f"validated tutorial {tutorial.id} requires {', '.join(missing)}"
                )

    return Catalog(
        schema_version=raw["schema_version"],
        sources=raw.get("sources", {}),
        routes=routes,
        tutorials=tutorials,
    )
