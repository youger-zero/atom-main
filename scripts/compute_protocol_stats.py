from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Ext-PGDP5K protocol statistics for paper tables.")
    parser.add_argument("--ext-root", required=True)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ext_root = Path(args.ext_root)
    stats = json.loads((ext_root / "stats.json").read_text(encoding="utf-8"))

    result: dict[str, object] = {
        "source_root": stats.get("source_root"),
        "relation_names": stats.get("relation_names", []),
        "splits": {},
    }

    total_pairs = 0
    total_positive = 0
    total_samples = 0

    for split in ("train", "val", "test"):
        records = json.loads((ext_root / split / "records.json").read_text(encoding="utf-8"))
        primitive_pairs = 0
        positive_relations = 0
        class_counts = {name: 0 for name in stats.get("relation_names", [])}
        for record in records:
            nodes = record.get("nodes", [])
            primitive_pairs += len(nodes) * max(len(nodes) - 1, 0)
            for edge in record.get("positive_edges", []):
                positive_relations += len(edge.get("relations", []))
                for relation_name in edge.get("relations", []):
                    class_counts[relation_name] = class_counts.get(relation_name, 0) + 1

        num_samples = len(records)
        avg_relations = positive_relations / max(num_samples, 1)
        result["splits"][split] = {
            "samples": num_samples,
            "primitive_pairs": primitive_pairs,
            "class_counts": class_counts,
            "avg_relations_per_sample": avg_relations,
        }
        total_pairs += primitive_pairs
        total_positive += positive_relations
        total_samples += num_samples

    result["total"] = {
        "samples": total_samples,
        "primitive_pairs": total_pairs,
        "avg_relations_per_sample": total_positive / max(total_samples, 1),
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output_path:
        Path(args.output_path).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
