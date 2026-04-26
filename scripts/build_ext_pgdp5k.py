from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

RELATIONS = (
    "intersect",
    "tangent",
    "parallel",
    "perpendicular",
    "bisect",
)

RELATION_TO_INDEX = {name: idx for idx, name in enumerate(RELATIONS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Ext-PGDP5K derived protocol artifacts.")
    parser.add_argument("--source-root", default="data/PGDP5K")
    parser.add_argument("--output-root", default="data/PGDP5K/Ext-PGDP5K")
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--audit-size", type=int, default=200)
    parser.add_argument("--audit-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    annotations_by_split = {
        split: json.loads((source_root / "annotations" / f"{split}.json").read_text(encoding="utf-8"))
        for split in ("train", "val", "test")
    }
    split_ids = {
        split: [
            line.strip()
            for line in (source_root / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for split in ("train", "val", "test")
    }

    per_split_records: dict[str, list[dict]] = defaultdict(list)
    stats = {
        "source_root": str(source_root.resolve()),
        "max_nodes": args.max_nodes,
        "knn_k": args.knn_k,
        "splits": {},
        "relation_names": list(RELATIONS),
    }
    global_class_counts = Counter()
    global_positive_edge_count = 0

    for split in ("train", "val", "test"):
        split_class_counts = Counter()
        split_positive_edge_count = 0
        split_node_count: list[int] = []

        for sample_id in split_ids[split]:
            annotation = annotations_by_split[split][sample_id]
            logic = load_json(source_root / "logic_forms" / f"{sample_id}.json")
            nodes, centers = build_nodes(annotation, max_nodes=args.max_nodes)
            relation_targets = derive_relation_targets(annotation, logic, nodes)
            semantic_text = build_semantic_text(annotation, logic)
            knn_indices = build_knn_indices(centers, args.knn_k)

            positive_edges = []
            class_counts = Counter()
            for left in range(len(nodes)):
                for right in range(left + 1, len(nodes)):
                    active = [
                        RELATIONS[rel_idx]
                        for rel_idx in range(len(RELATIONS))
                        if relation_targets[left][right][rel_idx] > 0
                    ]
                    if not active:
                        continue
                    positive_edges.append(
                        {
                            "left": left,
                            "right": right,
                            "left_id": nodes[left]["id"],
                            "right_id": nodes[right]["id"],
                            "relations": active,
                        }
                    )
                    for relation_name in active:
                        class_counts[relation_name] += 1

            split_class_counts.update(class_counts)
            global_class_counts.update(class_counts)
            split_positive_edge_count += len(positive_edges)
            global_positive_edge_count += len(positive_edges)
            split_node_count.append(len(nodes))

            per_split_records[split].append(
                {
                    "sample_id": sample_id,
                    "file_name": annotation["file_name"],
                    "width": annotation["width"],
                    "height": annotation["height"],
                    "num_nodes": len(nodes),
                    "nodes": [{"id": node["id"], "type": node["type"]} for node in nodes],
                    "semantic_text": semantic_text,
                    "positive_edges": positive_edges,
                    "class_counts": dict(class_counts),
                    "knn_indices": knn_indices,
                }
            )

        stats["splits"][split] = {
            "num_samples": len(split_ids[split]),
            "positive_edge_count": split_positive_edge_count,
            "class_counts": {name: split_class_counts.get(name, 0) for name in RELATIONS},
            "avg_num_nodes": round(sum(split_node_count) / max(len(split_node_count), 1), 4),
        }

    audit_ids = sample_audit_ids(split_ids, args.audit_size, args.audit_seed)

    for split, records in per_split_records.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        with (split_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats["global"] = {
        "num_samples": sum(len(ids) for ids in split_ids.values()),
        "positive_edge_count": global_positive_edge_count,
        "class_counts": {name: global_class_counts.get(name, 0) for name in RELATIONS},
        "active_relations": [name for name in RELATIONS if global_class_counts.get(name, 0) > 0],
        "zero_sample_relations": [name for name in RELATIONS if global_class_counts.get(name, 0) == 0],
        "audit_size": len(audit_ids),
        "audit_seed": args.audit_seed,
    }
    (output_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_stats_csv(output_root / "stats.csv", stats)
    (output_root / "audit_ids.txt").write_text("\n".join(audit_ids) + "\n", encoding="utf-8")
    write_audit_template(output_root / "audit_template.csv", audit_ids)
    (output_root / "README.txt").write_text(build_readme(args), encoding="utf-8")

    print(f"Built Ext-PGDP5K at {output_root}")
    print(f"Audit sample count: {len(audit_ids)}")


def build_nodes(annotation: dict, max_nodes: int) -> tuple[list[dict], list[tuple[float, float]]]:
    nodes: list[dict] = []
    centers: list[tuple[float, float]] = []

    for point in annotation["geos"].get("points", []):
        x, y = point["loc"][0]
        nodes.append({"id": point["id"], "type": "point"})
        centers.append((x, y))

    for line in annotation["geos"].get("lines", []):
        (x1, y1), (x2, y2) = line["loc"]
        nodes.append({"id": line["id"], "type": "line"})
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    for circle in annotation["geos"].get("circles", []):
        (cx, cy), _, _ = circle["loc"]
        nodes.append({"id": circle["id"], "type": "circle"})
        centers.append((cx, cy))

    return nodes[:max_nodes], centers[:max_nodes]


def build_knn_indices(centers: list[tuple[float, float]], k: int) -> list[list[int]]:
    num_nodes = len(centers)
    if num_nodes == 0:
        return []
    rows = []
    for i, (x1, y1) in enumerate(centers):
        distances = []
        for j, (x2, y2) in enumerate(centers):
            distances.append((math.dist((x1, y1), (x2, y2)), j))
        distances.sort(key=lambda item: item[0])
        neigh = [idx for _, idx in distances[: min(k, num_nodes)]]
        while len(neigh) < k:
            neigh.append(i)
        rows.append(neigh)
    return rows


def build_semantic_text(annotation: dict, logic: dict) -> str:
    symbol_tokens = []
    for symbol in annotation.get("symbols", []):
        content = symbol.get("text_content")
        if content and symbol.get("text_class") in {"point", "line", "circle", "angle", "arc", "degree"}:
            symbol_tokens.append(content)
    logic_tokens = [humanize_logic_form(form) for form in logic.get("diagram_logic_forms", [])]
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


def derive_relation_targets(annotation: dict, logic: dict, nodes: list[dict]) -> list[list[list[int]]]:
    targets = [[[0 for _ in RELATIONS] for _ in nodes] for _ in nodes]
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
    circle_name_to_node = match_logic_circles_to_nodes(logic, nodes)

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

    for bisector_name, side_name in infer_bisectors(raw_logic):
        set_pair_label(targets, line_name_to_node, bisector_name, side_name, "bisect")

    for line_name, circle_name in infer_tangencies(raw_logic):
        set_cross_type_label(targets, line_name_to_node, circle_name_to_node, line_name, circle_name, "tangent")

    return targets


def set_pair_label(targets: list, line_name_to_node: dict[str, int], left_name: str, right_name: str, relation_name: str) -> None:
    left_idx = line_name_to_node.get(left_name)
    right_idx = line_name_to_node.get(right_name)
    if left_idx is None or right_idx is None:
        return
    rel_idx = RELATION_TO_INDEX[relation_name]
    targets[left_idx][right_idx][rel_idx] = 1
    targets[right_idx][left_idx][rel_idx] = 1


def set_cross_type_label(
    targets: list,
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
    targets[left_idx][right_idx][rel_idx] = 1
    targets[right_idx][left_idx][rel_idx] = 1


def infer_bisectors(raw_logic: list[str]) -> list[tuple[str, str]]:
    results = []
    pattern = re.compile(
        r"Equals\(MeasureOf\(Angle\(([^,]+), ([^,]+), ([^)]+)\)\), MeasureOf\(Angle\(([^,]+), ([^,]+), ([^)]+)\)\)\)"
    )
    for form in raw_logic:
        if "Equals(MeasureOf(Angle" not in form:
            continue
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
    results = []
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


def match_logic_lines_to_nodes(annotation: dict, logic: dict, nodes: list[dict]) -> dict[str, int]:
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


def match_logic_circles_to_nodes(logic: dict, nodes: list[dict]) -> dict[str, int]:
    logic_circles = [canonical_circle_name(token) for token in logic.get("circle_instances", [])]
    circle_nodes = [node for node in nodes if node["type"] == "circle"]
    mapping: dict[str, int] = {}
    if len(logic_circles) == len(circle_nodes):
        for logic_name, node in zip(logic_circles, circle_nodes):
            mapping[logic_name] = node_index(nodes, node["id"], "circle")
    return mapping


def infer_annotation_line_names(annotation: dict) -> dict[str, str]:
    labelled_points = infer_point_labels(annotation)
    if not labelled_points:
        return infer_annotation_line_names_from_endpoints(annotation)

    result: dict[str, str] = {}
    for line in annotation["geos"].get("lines", []):
        matched = []
        for endpoint in line["loc"]:
            best = min(labelled_points, key=lambda item: math.dist(item[1], endpoint))
            matched.append(best[0])
        result[canonical_line_name(matched[0], matched[1])] = line["id"]
    return result


def infer_annotation_line_names_from_endpoints(annotation: dict) -> dict[str, str]:
    points = annotation["geos"].get("points", [])
    if not points:
        return {}
    result: dict[str, str] = {}
    for line in annotation["geos"].get("lines", []):
        matched = []
        for endpoint in line["loc"]:
            best_point = min(points, key=lambda point: math.dist(tuple(point["loc"][0]), endpoint))
            matched.append(best_point["id"])
        result[canonical_line_name(matched[0], matched[1])] = line["id"]
    return result


def infer_point_labels(annotation: dict) -> list[tuple[str, tuple[float, float], str]]:
    point_lookup = {point["id"]: tuple(point["loc"][0]) for point in annotation["geos"].get("points", [])}
    symbol_lookup = {symbol["id"]: symbol for symbol in annotation.get("symbols", [])}
    labelled_points = []
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


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deduplicate_preserve_order(items: list) -> list:
    seen = set()
    deduped = []
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


def node_index(nodes: list[dict], node_id: str, node_type: str) -> int:
    for idx, node in enumerate(nodes):
        if node["id"] == node_id and node["type"] == node_type:
            return idx
    raise KeyError(f"Missing node {node_type}:{node_id}")


def sample_audit_ids(split_ids: dict[str, list[str]], audit_size: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    all_ids = [sample_id for split in ("train", "val", "test") for sample_id in split_ids[split]]
    audit_size = min(audit_size, len(all_ids))
    return sorted(rng.sample(all_ids, audit_size))


def write_stats_csv(path: Path, stats: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "num_samples", "positive_edge_count", "avg_num_nodes", *RELATIONS])
        for split in ("train", "val", "test"):
            split_stats = stats["splits"][split]
            writer.writerow(
                [
                    split,
                    split_stats["num_samples"],
                    split_stats["positive_edge_count"],
                    split_stats["avg_num_nodes"],
                    *[split_stats["class_counts"].get(name, 0) for name in RELATIONS],
                ]
            )
        writer.writerow(
            [
                "global",
                stats["global"]["num_samples"],
                stats["global"]["positive_edge_count"],
                "",
                *[stats["global"]["class_counts"].get(name, 0) for name in RELATIONS],
            ]
        )


def build_readme(args: argparse.Namespace) -> str:
    return (
        "Ext-PGDP5K derived protocol artifacts\n"
        "=====================================\n\n"
        f"Source root: {args.source_root}\n"
        f"Output root: {args.output_root}\n"
        f"Max nodes: {args.max_nodes}\n"
        f"KNN k: {args.knn_k}\n"
        f"Audit size: {args.audit_size}\n"
        f"Audit seed: {args.audit_seed}\n\n"
        "Files:\n"
        "- stats.json / stats.csv: split-level protocol statistics\n"
        "- audit_ids.txt: fixed sample IDs for manual audit\n"
        "- audit_template.csv: manual audit template\n"
        "- <split>/records.json: per-sample derived artifacts\n"
        "- <split>/records.jsonl: line-delimited version for scripting\n"
    )


def write_audit_template(path: Path, audit_ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "derived_label_correct",
                "notes",
                "intersect_correct",
                "tangent_correct",
                "parallel_correct",
                "perpendicular_correct",
                "bisect_correct",
            ]
        )
        for sample_id in audit_ids:
            writer.writerow([sample_id, "", "", "", "", "", "", ""])


if __name__ == "__main__":
    main()
