from __future__ import annotations

RELATIONS = (
    "intersect",
    "tangent",
    "parallel",
    "perpendicular",
    "bisect",
)

ATOM_FAMILIES = (
    "parallel",
    "perpendicular",
    "tangent",
    "bisector",
    "angle_bisector",
    "intersection",
)

RELATION_TO_INDEX = {name: idx for idx, name in enumerate(RELATIONS)}
ATOM_TO_INDEX = {name: idx for idx, name in enumerate(ATOM_FAMILIES)}

UNDIRECTED_RELATIONS = ("parallel", "perpendicular", "intersect", "tangent")
TRANSITIVE_RELATIONS = ("parallel",)

DEFAULT_EXCLUSIVE_PAIRS = (
    ("parallel", "perpendicular"),
    ("parallel", "intersect"),
    ("parallel", "tangent"),
    ("perpendicular", "tangent"),
)


def normalize_relations(relation_names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if relation_names is None:
        return RELATIONS
    normalized = []
    seen = set()
    for name in relation_names:
        key = str(name).strip().lower()
        if not key:
            continue
        if key not in RELATION_TO_INDEX:
            raise ValueError(f"Unknown relation name: {name}")
        if key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    if not normalized:
        raise ValueError("At least one active relation is required.")
    return tuple(normalized)


def relation_indices(relation_names: list[str] | tuple[str, ...] | None) -> tuple[int, ...]:
    return tuple(RELATION_TO_INDEX[name] for name in normalize_relations(relation_names))


def filter_exclusive_pairs(
    relation_names: list[str] | tuple[str, ...] | None,
    exclusive_pairs: tuple[tuple[str, str], ...] = DEFAULT_EXCLUSIVE_PAIRS,
) -> tuple[tuple[str, str], ...]:
    active = set(normalize_relations(relation_names))
    return tuple((left, right) for left, right in exclusive_pairs if left in active and right in active)
