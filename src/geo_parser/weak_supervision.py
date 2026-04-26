from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .constants import ATOM_FAMILIES, ATOM_TO_INDEX


@dataclass(frozen=True)
class AtomicCueAnnotator:
    synonyms: Dict[str, Sequence[str]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "synonyms", self.synonyms or self.default_synonyms())
        compiled = {
            name: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for name, patterns in self.synonyms.items()
        }
        object.__setattr__(self, "_compiled", compiled)

    @staticmethod
    def default_synonyms() -> Dict[str, Sequence[str]]:
        return {
            "parallel": (r"\bparallel\b", r"\bis parallel to\b"),
            "perpendicular": (r"\bperpendicular\b", r"\borthogonal\b", r"\bnormal to\b"),
            "tangent": (r"\btangent\b", r"\btouches?\b"),
            "bisector": (r"\bbisect(?:s|ed|ing)?\b", r"\bmidpoint line\b"),
            "angle_bisector": (r"\bangle bisector\b", r"\bbisects? angle\b"),
            "intersection": (r"\bintersect(?:s|ed|ing)?\b", r"\bmeet(?:s|ing)? at\b"),
        }

    def encode(self, text: str) -> List[int]:
        normalized = self.normalize(text)
        return [int(name in normalized) for name in ATOM_FAMILIES]

    def normalize(self, text: str) -> List[str]:
        hits: List[str] = []
        for family, patterns in self._compiled.items():
            if any(pattern.search(text) for pattern in patterns):
                hits.append(family)
        return hits

    def encode_many(self, texts: Iterable[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def relation_prior(self, text: str) -> List[float]:
        atoms = self.encode(text)
        prior = [0.0] * len(ATOM_FAMILIES)
        for idx, active in enumerate(atoms):
            prior[idx] = float(active)
        return prior

    @staticmethod
    def active_indices(text: str) -> List[int]:
        annotator = AtomicCueAnnotator()
        return [ATOM_TO_INDEX[name] for name in annotator.normalize(text)]
