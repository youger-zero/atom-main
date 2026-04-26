# Ext-PGDP5K Protocol Notes

## Purpose

Ext-PGDP5K is a derived parser-level protocol built on top of PGDP5K.

The goal is to study text-conditioned geometric relation parsing with five nominal high-level relation labels:

1. `intersect`
2. `tangent`
3. `parallel`
4. `perpendicular`
5. `bisect`

The protocol is not a replacement for original PGDP5K. It is an author-defined extension for parser-level analysis.

In the current derived build, the active training labels are:

1. `intersect`
2. `parallel`
3. `perpendicular`
4. `bisect`

The `tangent` label is retained as a nominal relation for auditing, but the present derivation rules yield zero confirmed training labels. Tangent evidence should therefore be treated as an audit track until the derivation rule is strengthened.

## Source Signals

The derivation uses:

1. `annotations/<split>.json`
2. `logic_forms/<id>.json`
3. geometry symbols in annotations

## Derived Semantic Text

Each sample builds a semantic text string from:

1. geometry-relevant symbol text
2. humanized logic forms

This text is used only as a parser input representation and weak semantic supervision source.

## Derived Relation Rules

## Parallel

If a logic form contains:

`Parallel(Line(A, B), Line(C, D))`

then the corresponding line pair is labeled `parallel`.

## Perpendicular

If a logic form contains:

`Perpendicular(Line(A, B), Line(C, D))`

then the corresponding line pair is labeled `perpendicular`.

## Intersect

If two derived line instances share the same point through:

`PointLiesOnLine(P, Line(A, B))`

then the line pair is labeled `intersect`.

If a point lies on both a line and a circle, the line-circle pair is also labeled `intersect`.

## Tangent

A weak tangent label is derived when:

1. a point lies on a circle
2. a line involving that point participates in a perpendicular relation

This is a weak heuristic and should be manually audited.

## Bisect

A weak bisect label is derived from angle-equality patterns:

`Equals(MeasureOf(Angle(...)), MeasureOf(Angle(...)))`

When two equal angles share the same vertex and one common side, the shared side is treated as a bisector candidate.

This is also a weak heuristic and should be manually audited.

## Important Caveats

1. `parallel` and `perpendicular` are relatively direct from logic forms.
2. `intersect` is structurally derived.
3. `tangent` and `bisect` are weaker and more heuristic.
4. The protocol must therefore be accompanied by:
   - class statistics
   - manual audit
   - leakage-control baselines
5. The main reported experiments should use the active relation subset recorded in `stats.json`.

## Output Artifacts

The build script writes:

- `stats.json`
- `stats.csv`
- `audit_ids.txt`
- `audit_template.csv`
- `tangent_report.json`
- `<split>/records.json`
- `<split>/records.jsonl`

The training code can consume the derived records directly when `Ext-PGDP5K` exists under the PGDP5K root, or through an explicit `--ext-root` path when raw PGDP5K and derived protocol files live in different locations.
