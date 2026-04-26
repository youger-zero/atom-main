from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze potential tangent candidates in PGDP5K logic forms.")
    parser.add_argument("--source-root", default="data/PGDP5K")
    parser.add_argument("--report-path", default="data/PGDP5K/Ext-PGDP5K/tangent_report.json")
    parser.add_argument("--max-examples", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    split_ids = {
        split: [
            line.strip()
            for line in (source_root / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for split in ("train", "val", "test")
    }

    point_circle_pattern = re.compile(r"PointLiesOnCircle\(([^,]+), Circle\(([^)]+)\)\)")
    perp_pattern = re.compile(r"Perpendicular\(Line\(([^,]+), ([^)]+)\), Line\(([^,]+), ([^)]+)\)\)")

    split_stats = {}
    example_records = []

    for split in ("train", "val", "test"):
        counts = Counter()
        for sample_id in split_ids[split]:
            logic = json.loads((source_root / "logic_forms" / f"{sample_id}.json").read_text(encoding="utf-8"))
            forms = logic.get("diagram_logic_forms", [])
            point_to_circle = {}

            for form in forms:
                match = point_circle_pattern.match(form)
                if match:
                    point_to_circle.setdefault(match.group(1).strip(), set()).add(match.group(2).strip())

            has_circle = bool(logic.get("circle_instances"))
            has_point_on_circle = bool(point_to_circle)
            has_perp = False
            tangent_like_hits = []

            for form in forms:
                match = perp_pattern.match(form)
                if not match:
                    continue
                has_perp = True
                a, b, c, d = [token.strip() for token in match.groups()]
                shared_circle_points = [p for p in (a, b, c, d) if p in point_to_circle]
                if shared_circle_points:
                    tangent_like_hits.append(
                        {
                            "form": form,
                            "circle_points": sorted(shared_circle_points),
                            "circles": sorted({circle for point in shared_circle_points for circle in point_to_circle[point]}),
                        }
                    )

            if has_circle:
                counts["samples_with_circle"] += 1
            if has_point_on_circle:
                counts["samples_with_point_on_circle"] += 1
            if has_perp:
                counts["samples_with_perpendicular"] += 1
            if tangent_like_hits:
                counts["samples_with_tangent_like_pattern"] += 1
                if len(example_records) < args.max_examples:
                    example_records.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "circle_instances": logic.get("circle_instances", []),
                            "point_on_circle": {k: sorted(v) for k, v in point_to_circle.items()},
                            "tangent_like_hits": tangent_like_hits,
                            "logic_forms": forms,
                        }
                    )

        split_stats[split] = dict(counts)

    report = {
        "source_root": str(source_root.resolve()),
        "split_stats": split_stats,
        "example_count": len(example_records),
        "examples": example_records,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote tangent report to {report_path}")
    for split, stats in split_stats.items():
        print(split, stats)


if __name__ == "__main__":
    main()
