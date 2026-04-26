from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .constants import ATOM_FAMILIES, RELATION_TO_INDEX, RELATIONS, normalize_relations, relation_indices
from .weak_supervision import AtomicCueAnnotator


@dataclass
class Sample:
    sample_id: str
    text: str
    node_features: torch.Tensor
    relation_targets: torch.Tensor
    atomic_targets: torch.Tensor
    node_mask: torch.Tensor
    knn_indices: torch.Tensor


class PGDP5KDataset(Dataset[Sample]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        max_nodes: int = 64,
        knn_k: int = 8,
        active_relations: list[str] | tuple[str, ...] | None = None,
        ext_root: str | Path | None = None,
        shuffle_text: bool = False,
        shuffle_seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.max_nodes = max_nodes
        self.knn_k = knn_k
        self.active_relations = normalize_relations(active_relations)
        self.active_relation_indices = relation_indices(self.active_relations)
        self.annotator = AtomicCueAnnotator()
        self.logic_root = self.root / "logic_forms"
        explicit_ext_root = Path(ext_root) if ext_root is not None else None
        if explicit_ext_root is not None and explicit_ext_root.exists():
            self.ext_root = explicit_ext_root
        else:
            self.ext_root = self.root / "Ext-PGDP5K" if (self.root / "Ext-PGDP5K").exists() else None

        annotation_path = self.root / "annotations" / f"{split}.json"
        split_path = self.root / "splits" / f"{split}.txt"

        self.annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        self.ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.ext_records = load_ext_records(self.ext_root, split) if self.ext_root is not None else {}
        self._class_balance_stats: dict[str, list[float]] | None = None
        self.shuffle_text = shuffle_text
        self.shuffle_seed = shuffle_seed
        self.shuffled_texts = self._build_shuffled_texts() if shuffle_text else {}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Sample:
        sample_id = self.ids[index]
        annotation = self.annotations[sample_id]
        logic = load_json(self.logic_root / f"{sample_id}.json")

        nodes, centers = build_nodes(annotation, max_nodes=self.max_nodes)
        ext_record = self.ext_records.get(sample_id)
        text = ext_record["semantic_text"] if ext_record is not None else build_semantic_text(annotation, logic)
        if self.shuffle_text:
            text = self.shuffled_texts[sample_id]
        relation_targets = targets_from_ext_record(ext_record, nodes) if ext_record is not None else derive_relation_targets(annotation, logic, nodes)
        relation_targets = relation_targets[..., list(self.active_relation_indices)]
        node_features = torch.tensor([node["feature"] for node in nodes], dtype=torch.float32)
        atomic_targets = torch.tensor(self.annotator.encode(text), dtype=torch.float32)
        node_mask = torch.ones(len(nodes), dtype=torch.bool)
        knn_indices = torch.tensor(ext_record["knn_indices"], dtype=torch.long) if ext_record is not None else build_knn_indices(centers, self.knn_k)

        return Sample(
            sample_id=sample_id,
            text=text,
            node_features=node_features,
            relation_targets=relation_targets,
            atomic_targets=atomic_targets,
            node_mask=node_mask,
            knn_indices=knn_indices,
        )

    def _build_shuffled_texts(self) -> dict[str, str]:
        text_by_id: dict[str, str] = {}
        for sample_id in self.ids:
            annotation = self.annotations[sample_id]
            logic = load_json(self.logic_root / f"{sample_id}.json")
            ext_record = self.ext_records.get(sample_id)
            text_by_id[sample_id] = ext_record["semantic_text"] if ext_record is not None else build_semantic_text(annotation, logic)

        shuffled_ids = list(self.ids)
        rng = random.Random(self.shuffle_seed)
        if len(shuffled_ids) > 1:
            while True:
                rng.shuffle(shuffled_ids)
                if all(left != right for left, right in zip(self.ids, shuffled_ids)):
                    break

        return {
            sample_id: text_by_id[shuffled_id]
            for sample_id, shuffled_id in zip(self.ids, shuffled_ids)
        }

    def class_balance_stats(self) -> dict[str, list[float]]:
        if self._class_balance_stats is not None:
            return self._class_balance_stats

        positive_counts = [0.0 for _ in self.active_relations]
        valid_counts = [0.0 for _ in self.active_relations]

        records = self.ext_records.values() if self.ext_records else []
        for record in records:
            nodes = record.get("nodes", [])
            line_count = sum(1 for node in nodes if node.get("type") == "line")
            circle_count = sum(1 for node in nodes if node.get("type") == "circle")

            line_line = float(line_count * max(line_count - 1, 0))
            line_circle = float(2 * line_count * circle_count)

            for rel_idx, relation_name in enumerate(self.active_relations):
                if relation_name == "intersect":
                    valid_counts[rel_idx] += line_line + line_circle
                elif relation_name in {"parallel", "perpendicular", "bisect"}:
                    valid_counts[rel_idx] += line_line
                elif relation_name == "tangent":
                    valid_counts[rel_idx] += line_circle

            for edge in record.get("positive_edges", []):
                for relation_name in edge.get("relations", []):
                    if relation_name not in self.active_relations:
                        continue
                    rel_idx = self.active_relations.index(relation_name)
                    positive_counts[rel_idx] += 2.0

        pos_weight = []
        for pos, valid in zip(positive_counts, valid_counts):
            neg = max(valid - pos, 1.0)
            weight = neg / max(pos, 1.0)
            pos_weight.append(min(max(weight ** 0.5, 1.0), 4.0))

        self._class_balance_stats = {
            "positive_counts": positive_counts,
            "valid_counts": valid_counts,
            "pos_weight": pos_weight,
        }
        return self._class_balance_stats


def collate_samples(samples: list[Sample]) -> dict[str, Any]:
    batch_size = len(samples)
    max_nodes = max(sample.node_features.shape[0] for sample in samples)
    feature_dim = samples[0].node_features.shape[-1]
    num_relations = samples[0].relation_targets.shape[-1]
    max_knn = max(sample.knn_indices.shape[-1] for sample in samples)

    node_features = torch.zeros(batch_size, max_nodes, feature_dim)
    relation_targets = torch.zeros(batch_size, max_nodes, max_nodes, num_relations)
    atomic_targets = torch.zeros(batch_size, len(ATOM_FAMILIES))
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    knn_indices = torch.zeros(batch_size, max_nodes, max_knn, dtype=torch.long)
    texts: list[str] = []
    sample_ids: list[str] = []

    for row, sample in enumerate(samples):
        n = sample.node_features.shape[0]
        node_features[row, :n] = sample.node_features
        relation_targets[row, :n, :n] = sample.relation_targets
        atomic_targets[row] = sample.atomic_targets
        node_mask[row, :n] = sample.node_mask
        knn_indices[row, :n, : sample.knn_indices.shape[-1]] = sample.knn_indices
        texts.append(sample.text)
        sample_ids.append(sample.sample_id)

    return {
        "sample_ids": sample_ids,
        "texts": texts,
        "node_features": node_features,
        "relation_targets": relation_targets,
        "atomic_targets": atomic_targets,
        "node_mask": node_mask,
        "knn_indices": knn_indices,
    }


def build_nodes(annotation: dict[str, Any], max_nodes: int) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
    width = max(float(annotation["width"]), 1.0)
    height = max(float(annotation["height"]), 1.0)
    diag = math.sqrt(width * width + height * height)

    nodes: list[dict[str, Any]] = []
    centers: list[tuple[float, float]] = []

    for point in annotation["geos"].get("points", []):
        x, y = point["loc"][0]
        feature = [1.0, 0.0, 0.0, x / width, y / height, 0.0, 0.0, 0.0, x / width, y / height, x / width, y / height]
        nodes.append({"id": point["id"], "type": "point", "feature": feature})
        centers.append((x, y))

    for line in annotation["geos"].get("lines", []):
        (x1, y1), (x2, y2) = line["loc"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = (x2 - x1) / width
        dy = (y2 - y1) / height
        length = math.dist((x1, y1), (x2, y2)) / diag
        feature = [0.0, 1.0, 0.0, cx / width, cy / height, dx, dy, length, x1 / width, y1 / height, x2 / width, y2 / height]
        nodes.append({"id": line["id"], "type": "line", "feature": feature})
        centers.append((cx, cy))

    for circle in annotation["geos"].get("circles", []):
        (cx, cy), radius, _ = circle["loc"]
        feature = [0.0, 0.0, 1.0, cx / width, cy / height, radius / diag, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        nodes.append({"id": circle["id"], "type": "circle", "feature": feature})
        centers.append((cx, cy))

    nodes = nodes[:max_nodes]
    centers = centers[:max_nodes]
    return nodes, centers


def build_knn_indices(centers: list[tuple[float, float]], k: int) -> torch.Tensor:
    num_nodes = len(centers)
    if num_nodes == 0:
        return torch.zeros(0, 0, dtype=torch.long)
    rows = []
    for i, (x1, y1) in enumerate(centers):
        distances = []
        for j, (x2, y2) in enumerate(centers):
            dist = math.dist((x1, y1), (x2, y2))
            distances.append((dist, j))
        distances.sort(key=lambda item: item[0])
        neigh = [idx for _, idx in distances[: min(k, num_nodes)]]
        while len(neigh) < k:
            neigh.append(i)
        rows.append(neigh)
    return torch.tensor(rows, dtype=torch.long)


def build_semantic_text(annotation: dict[str, Any], logic: dict[str, Any]) -> str:
    symbol_tokens = []
    for symbol in annotation.get("symbols", []):
        content = symbol.get("text_content")
        if content and symbol.get("text_class") in {"point", "line", "circle", "angle", "arc", "degree"}:
            symbol_tokens.append(content)
    logic_tokens = []
    for form in logic.get("diagram_logic_forms", []):
        logic_tokens.append(humanize_logic_form(form))
    return " ".join(deduplicate_preserve_order(symbol_tokens + logic_tokens))


def humanize_logic_form(form: str) -> str:
    replacements = {
        "PointLiesOnLine": "point on line",
        "PointLiesOnCircle": "point on circle",
        "Perpendicular": "perpendicular",
        "Parallel": "parallel",
        "Equals": "equals",
        "MeasureOf": "angle",
        "LengthOf": "length",
        "Line": "line",
        "Circle": "circle",
        "Angle": "angle",
    }
    text = form
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()


def derive_relation_targets(
    annotation: dict[str, Any],
    logic: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> torch.Tensor:
    targets = torch.zeros(len(nodes), len(nodes), len(RELATIONS), dtype=torch.float32)
    point_to_lines: dict[str, set[str]] = {}
    point_to_circles: dict[str, set[str]] = {}
    raw_logic = logic.get("diagram_logic_forms", [])

    for form in raw_logic:
        match = re.match(r"PointLiesOnLine\(([^,]+), Line\(([^,]+), ([^)]+)\)\)", form)
        if match:
            point_name = match.group(1).strip()
            line_name = canonical_line_name(match.group(2), match.group(3))
            point_to_lines.setdefault(point_name, set()).add(line_name)
            continue
        match = re.match(r"PointLiesOnCircle\(([^,]+), Circle\(([^)]+)\)\)", form)
        if match:
            point_name = match.group(1).strip()
            circle_name = canonical_circle_name(match.group(2))
            point_to_circles.setdefault(point_name, set()).add(circle_name)

    line_name_to_node = match_logic_lines_to_nodes(annotation, logic, nodes)
    circle_name_to_node = match_logic_circles_to_nodes(annotation, logic, nodes)

    for form in raw_logic:
        match = re.match(r"Parallel\(Line\(([^,]+), ([^)]+)\), Line\(([^,]+), ([^)]+)\)\)", form)
        if match:
            set_pair_label(targets, line_name_to_node, canonical_line_name(match.group(1), match.group(2)), canonical_line_name(match.group(3), match.group(4)), "parallel")
            continue
        match = re.match(r"Perpendicular\(Line\(([^,]+), ([^)]+)\), Line\(([^,]+), ([^)]+)\)\)", form)
        if match:
            set_pair_label(targets, line_name_to_node, canonical_line_name(match.group(1), match.group(2)), canonical_line_name(match.group(3), match.group(4)), "perpendicular")

    for point_name, line_names in point_to_lines.items():
        line_list = sorted(line_names)
        for i in range(len(line_list)):
            for j in range(i + 1, len(line_list)):
                set_pair_label(targets, line_name_to_node, line_list[i], line_list[j], "intersect")

        for circle_name in sorted(point_to_circles.get(point_name, [])):
            for line_name in line_list:
                set_cross_type_label(targets, line_name_to_node, circle_name_to_node, line_name, circle_name, "intersect")

    bisectors = infer_bisectors(raw_logic)
    for bisector_name, side_name in bisectors:
        set_pair_label(targets, line_name_to_node, bisector_name, side_name, "bisect")

    tangencies = infer_tangencies(raw_logic)
    for line_name, circle_name in tangencies:
        set_cross_type_label(targets, line_name_to_node, circle_name_to_node, line_name, circle_name, "tangent")

    return targets


def set_pair_label(
    targets: torch.Tensor,
    line_name_to_node: dict[str, int],
    left_name: str,
    right_name: str,
    relation_name: str,
) -> None:
    left_idx = line_name_to_node.get(left_name)
    right_idx = line_name_to_node.get(right_name)
    if left_idx is None or right_idx is None:
        return
    rel_idx = RELATION_TO_INDEX[relation_name]
    targets[left_idx, right_idx, rel_idx] = 1.0
    targets[right_idx, left_idx, rel_idx] = 1.0


def set_cross_type_label(
    targets: torch.Tensor,
    line_name_to_node: dict[str, int],
    circle_name_to_node: dict[str, int],
    line_name: str,
    circle_name: str,
    relation_name: str,
) -> None:
    left_idx = line_name_to_node.get(line_name)
    right_idx = circle_name_to_node.get(circle_name)
    if left_idx is None or right_idx is None:
        return
    rel_idx = RELATION_TO_INDEX[relation_name]
    targets[left_idx, right_idx, rel_idx] = 1.0
    targets[right_idx, left_idx, rel_idx] = 1.0


def infer_bisectors(raw_logic: list[str]) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    angle_equalities = [form for form in raw_logic if "Equals(MeasureOf(Angle" in form]
    pattern = re.compile(
        r"Equals\(MeasureOf\(Angle\(([^,]+), ([^,]+), ([^)]+)\)\), MeasureOf\(Angle\(([^,]+), ([^,]+), ([^)]+)\)\)\)"
    )
    for form in angle_equalities:
        match = pattern.match(form)
        if not match:
            continue
        a1, v1, b1, a2, v2, b2 = [token.strip() for token in match.groups()]
        if v1 != v2:
            continue
        shared = sorted(set([a1, b1]).intersection([a2, b2]))
        if len(shared) != 1:
            continue
        shared_point = shared[0]
        others = sorted({p for p in [a1, b1, a2, b2] if p != shared_point})
        if len(others) < 2:
            continue
        bisector = canonical_line_name(v1, shared_point)
        for other in others:
            results.append((bisector, canonical_line_name(v1, other)))
    return results


def infer_tangencies(raw_logic: list[str]) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    perp_pattern = re.compile(r"Perpendicular\(Line\(([^,]+), ([^)]+)\), Line\(([^,]+), ([^)]+)\)\)")
    point_circle_pattern = re.compile(r"PointLiesOnCircle\(([^,]+), Circle\(([^)]+)\)\)")
    point_to_circle: dict[str, set[str]] = {}
    for form in raw_logic:
        match = point_circle_pattern.match(form)
        if match:
            point_to_circle.setdefault(match.group(1).strip(), set()).add(canonical_circle_name(match.group(2)))

    for form in raw_logic:
        match = perp_pattern.match(form)
        if not match:
            continue
        a, b, c, d = [token.strip() for token in match.groups()]
        if a in point_to_circle and b not in point_to_circle:
            for circle_name in sorted(point_to_circle[a]):
                results.append((canonical_line_name(a, b), circle_name))
        if b in point_to_circle and a not in point_to_circle:
            for circle_name in sorted(point_to_circle[b]):
                results.append((canonical_line_name(a, b), circle_name))
        if c in point_to_circle and d not in point_to_circle:
            for circle_name in sorted(point_to_circle[c]):
                results.append((canonical_line_name(c, d), circle_name))
        if d in point_to_circle and c not in point_to_circle:
            for circle_name in sorted(point_to_circle[d]):
                results.append((canonical_line_name(c, d), circle_name))
    return deduplicate_preserve_order(results)


def match_logic_lines_to_nodes(
    annotation: dict[str, Any],
    logic: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> dict[str, int]:
    logic_lines = [canonical_line_name_from_token(token) for token in logic.get("line_instances", [])]
    line_nodes = [node for node in nodes if node["type"] == "line"]
    mapping: dict[str, int] = {}

    annotation_line_names = infer_annotation_line_names(annotation)
    for logic_name in logic_lines:
        if logic_name in annotation_line_names:
            node_id = annotation_line_names[logic_name]
            mapping[logic_name] = node_index(nodes, node_id, "line")

    unmatched_logic = [name for name in logic_lines if name not in mapping]
    used_node_ids = {nodes[idx]["id"] for idx in mapping.values()}
    remaining_nodes = [node for node in line_nodes if node["id"] not in used_node_ids]
    if len(unmatched_logic) == len(remaining_nodes):
        for logic_name, node in zip(unmatched_logic, remaining_nodes):
            mapping[logic_name] = node_index(nodes, node["id"], "line")
    return mapping


def match_logic_circles_to_nodes(
    annotation: dict[str, Any],
    logic: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> dict[str, int]:
    logic_circles = [canonical_circle_name(token) for token in logic.get("circle_instances", [])]
    circle_nodes = [node for node in nodes if node["type"] == "circle"]
    mapping: dict[str, int] = {}
    if len(logic_circles) == len(circle_nodes):
        for logic_name, node in zip(logic_circles, circle_nodes):
            mapping[logic_name] = node_index(nodes, node["id"], "circle")
    return mapping


def infer_annotation_line_names(annotation: dict[str, Any]) -> dict[str, str]:
    labelled_points = infer_point_labels(annotation)
    if not labelled_points:
        return infer_annotation_line_names_from_endpoints(annotation)

    result: dict[str, str] = {}
    for line in annotation["geos"].get("lines", []):
        endpoints = line["loc"]
        matched = []
        for endpoint in endpoints:
            best = min(labelled_points, key=lambda item: math.dist(item[1], endpoint))
            matched.append(best[0])
        result[canonical_line_name(matched[0], matched[1])] = line["id"]
    return result


def infer_annotation_line_names_from_endpoints(annotation: dict[str, Any]) -> dict[str, str]:
    point_ids = [point["id"] for point in annotation["geos"].get("points", [])]
    if not point_ids:
        return {}
    result: dict[str, str] = {}
    for line in annotation["geos"].get("lines", []):
        endpoints = line["loc"]
        matched = []
        for endpoint in endpoints:
            best_point = min(
                annotation["geos"].get("points", []),
                key=lambda point: math.dist(tuple(point["loc"][0]), endpoint),
            )
            matched.append(best_point["id"])
        result[canonical_line_name(matched[0], matched[1])] = line["id"]
    return result


def infer_point_labels(annotation: dict[str, Any]) -> list[tuple[str, tuple[float, float], str]]:
    point_lookup = {point["id"]: tuple(point["loc"][0]) for point in annotation["geos"].get("points", [])}
    symbol_lookup = {symbol["id"]: symbol for symbol in annotation.get("symbols", [])}
    labelled_points: list[tuple[str, tuple[float, float], str]] = []
    used_point_ids: set[str] = set()

    for symbol_id, geo_ids in annotation.get("relations", {}).get("sym2geo", []):
        symbol = symbol_lookup.get(symbol_id)
        if not symbol or symbol.get("text_class") != "point":
            continue
        for geo_id in geo_ids:
            if geo_id in point_lookup and geo_id not in used_point_ids:
                labelled_points.append((symbol["text_content"], point_lookup[geo_id], geo_id))
                used_point_ids.add(geo_id)

    if labelled_points:
        return labelled_points

    for symbol in annotation.get("symbols", []):
        if symbol.get("text_class") != "point":
            continue
        center = bbox_center(symbol.get("bbox"))
        if center is None:
            continue
        point_id, point_loc = min(point_lookup.items(), key=lambda item: math.dist(item[1], center))
        if point_id in used_point_ids:
            continue
        labelled_points.append((symbol["text_content"], point_loc, point_id))
        used_point_ids.add(point_id)
    return labelled_points


def bbox_center(bbox: list[float] | None) -> tuple[float, float] | None:
    if not bbox or len(bbox) < 4:
        return None
    x, y, w, h = bbox[:4]
    return (x + w / 2.0, y + h / 2.0)


@lru_cache(maxsize=8192)
def load_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    return json.loads(file_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def load_ext_records(root: str | Path | None, split: str) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}
    root_path = Path(root)
    records_path = root_path / split / "records.json"
    if not records_path.exists():
        return {}
    records = json.loads(records_path.read_text(encoding="utf-8"))
    return {record["sample_id"]: record for record in records}


def targets_from_ext_record(record: dict[str, Any], nodes: list[dict[str, Any]]) -> torch.Tensor:
    targets = torch.zeros(len(nodes), len(nodes), len(RELATIONS), dtype=torch.float32)
    if record is None:
        return targets
    node_ids = [node["id"] for node in nodes]
    for edge in record.get("positive_edges", []):
        left = edge["left"]
        right = edge["right"]
        if left >= len(nodes) or right >= len(nodes):
            continue
        if edge.get("left_id") != node_ids[left] or edge.get("right_id") != node_ids[right]:
            continue
        for relation_name in edge.get("relations", []):
            rel_idx = RELATION_TO_INDEX[relation_name]
            targets[left, right, rel_idx] = 1.0
            targets[right, left, rel_idx] = 1.0
    return targets


def deduplicate_preserve_order(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    deduped: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def canonical_line_name(a: str, b: str) -> str:
    return "".join(sorted([a.strip(), b.strip()]))


def canonical_line_name_from_token(token: str) -> str:
    token = token.strip()
    if len(token) >= 2:
        return canonical_line_name(token[0], token[1])
    return token


def canonical_circle_name(token: str) -> str:
    return token.strip().upper()


def node_index(nodes: list[dict[str, Any]], node_id: str, node_type: str) -> int:
    for idx, node in enumerate(nodes):
        if node["id"] == node_id and node["type"] == node_type:
            return idx
    raise KeyError(f"Missing node {node_type}:{node_id}")
