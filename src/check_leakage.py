"""Validate the household electricity data split for common leakage risks.

Run from the repository root with ``python src/check_leakage.py``. The check
is intentionally a data contract check: it does not train a model or infer
leakage from correlations.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def date_bounds(rows: list[dict[str, str]], column: str) -> tuple[str, str]:
    values = [row[column] for row in rows if row.get(column)]
    if not values:
        raise ValueError(f"No values found in date column {column!r}")
    # ISO-8601 YYYY-MM-DD dates sort lexicographically in chronological order.
    return min(values), max(values)


def check(config: dict[str, Any]) -> dict[str, Any]:
    paths = {name: ROOT / value for name, value in config["data"].items()}
    tables = {name: read_csv(path) for name, path in paths.items()}
    target = config["target"]
    id_column = config["id_column"]
    date_column = config["date_column"]
    features = config["features"]
    forbidden = set(config["forbidden_features"])

    train, later, holdout = tables["train"], tables["later"], tables["holdout"]
    schemas = {name: set(rows[0]) if rows else set() for name, rows in tables.items()}
    errors: list[str] = []
    notes: list[str] = []

    for name, rows in tables.items():
        if not rows:
            errors.append(f"{name} dataset is empty")
        if id_column not in schemas[name]:
            errors.append(f"{name} is missing identifier column {id_column!r}")
        elif len({row[id_column] for row in rows}) != len(rows):
            errors.append(f"{name} contains duplicate {id_column} values")
        if date_column not in schemas[name]:
            errors.append(f"{name} is missing date column {date_column!r}")

    for name in ("train", "later"):
        missing = sorted(({target, *features} - schemas[name]))
        if missing:
            errors.append(f"{name} is missing required columns: {', '.join(missing)}")
    missing_holdout = sorted(set(features) - schemas["holdout"])
    if missing_holdout:
        errors.append("holdout is missing model features: " + ", ".join(missing_holdout))
    if target in schemas["holdout"]:
        errors.append(f"holdout contains target {target!r}; verify labels are not exposed")

    bad_features = sorted(set(features) & (forbidden | {target, id_column}))
    if bad_features:
        errors.append("configured model features include forbidden columns: " + ", ".join(bad_features))

    id_sets = {
        name: {row[id_column] for row in rows if id_column in row}
        for name, rows in tables.items()
    }
    for left, right in (("train", "later"), ("train", "holdout"), ("later", "holdout")):
        overlap = id_sets[left] & id_sets[right]
        if overlap:
            errors.append(f"{left} and {right} share {len(overlap)} record IDs")

    bounds: dict[str, dict[str, str]] = {}
    for name, rows in tables.items():
        if rows and date_column in schemas[name]:
            low, high = date_bounds(rows, date_column)
            bounds[name] = {"start": low, "end": high}
    if all(name in bounds for name in ("train", "later", "holdout")):
        if not (bounds["train"]["end"] < bounds["later"]["start"] < bounds["holdout"]["start"]):
            errors.append("date windows are not strictly ordered train < later < holdout")
        else:
            notes.append("chronological split ordering is valid")

    present_forbidden = sorted(forbidden & schemas["train"])
    if present_forbidden:
        notes.append(
            "excluded post-outcome/identifier fields are present in raw training data: "
            + ", ".join(present_forbidden)
        )

    return {
        "status": "pass" if not errors else "fail",
        "datasets": {
            name: {"rows": len(rows), "date_range": bounds.get(name)}
            for name, rows in tables.items()
        },
        "target": target,
        "model_features": features,
        "excluded_columns": sorted(forbidden | {target, id_column}),
        "checks": {
            "target_absent_from_holdout": target not in schemas["holdout"],
            "record_ids_unique_and_disjoint": not any("record ID" in e or "duplicate" in e for e in errors),
            "chronological_splits": any("chronological split ordering" in n for n in notes),
            "configured_features_safe": not bad_features,
        },
        "notes": notes,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "params.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "findings.yaml")
    args = parser.parse_args()

    try:
        with args.config.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        result = check(config)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Leakage check could not run: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(result, stream, sort_keys=False, allow_unicode=True)
    print(f"Leakage check: {result['status']} ({args.output})")
    for item in result["notes"]:
        print(f"- {item}")
    for item in result["errors"]:
        print(f"ERROR: {item}", file=sys.stderr)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
