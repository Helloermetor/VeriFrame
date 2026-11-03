"""
VeriFrame source-disjoint score audit
Read-only analysis of source_disjoint_results.csv.
Does NOT fit calibrators, change thresholds, or modify project files.
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "validation" / "robustness" / "source_disjoint" / "api_results" / "source_disjoint_results.csv"
OUT_DIR = CSV_PATH.parent / "calibration_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLUMNS = {
    "GenD": ("GenD", "class1_manipulated_score_mean"),
    "Xception": ("Xception", "class1_manipulated_score_mean"),
    "Effort": ("Effort", "class1_manipulated_score_mean"),
}
FAKESTORMER_FIELDS = {
    "temporal_final": ("temporal_localization_final",),
    "classification_head": ("classification_head_values",),
}


def parse_json_cell(value, row_num, column):
    try:
        return json.loads(value) if value else {}
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in row {row_num}, column {column}: {exc}"
        ) from exc


def finite_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def score_metrics(y_true, scores):
    """Optional sklearn metrics; return None if sklearn is unavailable or one class only."""
    try:
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            brier_score_loss,
            log_loss,
            confusion_matrix,
        )

        if len(set(y_true)) < 2:
            return {"note": "Only one class present; ranking metrics undefined."}
        pred = [int(s >= 0.5) for s in scores]
        cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
        clipped = [min(max(s, 1e-7), 1 - 1e-7) for s in scores]
        return {
            "n": len(y_true),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "average_precision": float(average_precision_score(y_true, scores)),
            "accuracy_at_0_5": float(accuracy_score(y_true, pred)),
            "precision_at_0_5": float(precision_score(y_true, pred, zero_division=0)),
            "recall_at_0_5": float(recall_score(y_true, pred, zero_division=0)),
            "f1_at_0_5": float(f1_score(y_true, pred, zero_division=0)),
            "brier_raw_score_diagnostic_only": float(brier_score_loss(y_true, scores)),
            "log_loss_raw_score_diagnostic_only": float(
                log_loss(y_true, clipped, labels=[0, 1])
            ),
            "confusion_matrix_labels_0_real_1_fake": cm,
        }
    except ImportError:
        return {"note": "scikit-learn unavailable; install it to calculate metrics."}


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("CSV contains no data rows.")

    required = {"Source_ID", "Class", "Transformation", "Status", *MODEL_COLUMNS.keys()}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    print("=" * 72)
    print("VERIFRAME SOURCE-DISJOINT SCORE AUDIT (READ ONLY)")
    print("=" * 72)
    print(f"Rows: {len(rows)}")
    print(f"Distinct Source_ID values: {len(set(r['Source_ID'] for r in rows))}")
    print("Status counts:", dict(Counter(r["Status"] for r in rows)))
    print("Class counts:", dict(Counter(r["Class"].lower() for r in rows)))
    print(
        "Transformation counts:",
        dict(Counter(r["Transformation"].lower() for r in rows)),
    )
    print(
        "Rows per Source_ID:",
        dict(sorted(Counter(r["Source_ID"] for r in rows).items())),
    )
    print()

    # Ensure each Source_ID contains the expected class/transformation cells.
    expected = {
        (cls, trans)
        for cls in ("real", "fake")
        for trans in ("baseline", "crf32", "240p")
    }
    coverage = defaultdict(set)
    for r in rows:
        label = r["Class"].lower()
        label = "real" if label == "original" else label

        coverage[r["Source_ID"]].add((label, r["Transformation"].lower()))
    incomplete = {
        sid: sorted(expected - cells)
        for sid, cells in coverage.items()
        if expected - cells
    }
    print(
        f"Source_ID groups with all 6 class/transformation combinations: "
        f"{len(coverage) - len(incomplete)}/{len(coverage)}"
    )
    if incomplete:
        print("Incomplete groups:")
        for sid, missing_cells in sorted(incomplete.items()):
            print(f"  {sid}: missing {missing_cells}")
    print()

    report = {
        "input_csv": str(CSV_PATH),
        "row_count": len(rows),
        "distinct_source_ids": len(coverage),
        "status_counts": dict(Counter(r["Status"] for r in rows)),
        "class_counts": dict(Counter(r["Class"].lower() for r in rows)),
        "transformation_counts": dict(
            Counter(r["Transformation"].lower() for r in rows)
        ),
        "complete_six_cell_source_groups": len(coverage) - len(incomplete),
        "incomplete_source_groups": {
            k: [list(x) for x in v] for k, v in incomplete.items()
        },
        "metrics": {},
        "score_ranges": {},
        "warnings": [
            "Raw Brier score and log loss are diagnostics only; raw model scores are not calibrated probabilities.",
            "The 20 Source_ID groups are repeated paired/transformed observations, not 120 independent videos.",
            "Source_ID grouping must be checked against the manifest's original/fake source identifiers before any split.",
            "No calibrator or threshold is fitted by this script.",
        ],
    }

    # Parse nested detector outputs and evaluate each transformation independently.
    for model, (column, score_key) in MODEL_COLUMNS.items():
        parsed = []
        for i, r in enumerate(rows, start=2):
            obj = parse_json_cell(r[column], i, column)
            score = finite_float(obj.get(score_key))
            parsed.append((r, obj, score))

        report["metrics"][model] = {}
        report["score_ranges"][model] = {}
        print(f"{model}:")
        for trans in ("baseline", "crf32", "240p"):
            subset = [
                (r, obj, s)
                for r, obj, s in parsed
                if r["Transformation"].lower() == trans
                and s is not None
                and r["Class"].lower() in ("real", "original", "fake", "manipulated")
            ]
            y = [
                1 if r["Class"].lower() in ("fake", "manipulated") else 0
                for r, _, _ in subset
            ]
            scores = [s for _, _, s in subset]
            metrics = (
                score_metrics(y, scores) if subset else {"note": "No valid scores."}
            )
            report["metrics"][model][trans] = metrics
            if scores:
                report["score_ranges"][model][trans] = {
                    "n": len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "mean": sum(scores) / len(scores),
                }
            print(f"  {trans}: {metrics}")
        print()

    # FakeSTormer is kept separate: sigmoid temporal/head outputs are not class-1 softmax scores.
    fs_summary = defaultdict(list)
    for i, r in enumerate(rows, start=2):
        obj = parse_json_cell(r["FakeSTormer"], i, "FakeSTormer")
        for key in ("temporal_localization_final",):
            value = finite_float(obj.get(key))
            if value is not None:
                fs_summary[key].append(value)
        head = obj.get("classification_head_values")
        if isinstance(head, list) and head:
            value = finite_float(head[0])
            if value is not None:
                fs_summary["classification_head_first_value"].append(value)

    report["fakestormer_score_ranges"] = {
        key: {
            "n": len(vals),
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
        }
        for key, vals in fs_summary.items()
        if vals
    }
    print("FakeSTormer (separate diagnostics; no pooled class-score metrics):")
    print(report["fakestormer_score_ranges"])
    print()

    # Check whether model outputs are missing/failed.
    for model in ("GenD", "Xception", "Effort", "FakeSTormer"):
        statuses = Counter()
        for i, r in enumerate(rows, start=2):
            obj = parse_json_cell(r[model], i, model)
            statuses[str(obj.get("status", "missing"))] += 1
        print(f"{model} nested status counts: {dict(statuses)}")

    json_path = OUT_DIR / "score_audit_summary.json"
    csv_path = OUT_DIR / "score_audit_by_group.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Export per-source/per-transformation score rows for transparent inspection.
    output_rows = []
    for i, r in enumerate(rows, start=2):
        out = {
            "Source_ID": r["Source_ID"],
            "Class": r["Class"],
            "Transformation": r["Transformation"],
            "Filename": r["Filename"],
            "Status": r["Status"],
        }
        for model, (column, score_key) in MODEL_COLUMNS.items():
            obj = parse_json_cell(r[column], i, column)
            out[f"{model}_raw_score"] = obj.get(score_key, "")
        fs = parse_json_cell(r["FakeSTormer"], i, "FakeSTormer")
        out["FakeSTormer_temporal_final"] = fs.get("temporal_localization_final", "")
        head = fs.get("classification_head_values")
        out["FakeSTormer_classification_head_first"] = (
            head[0] if isinstance(head, list) and head else ""
        )
        output_rows.append(out)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    print()
    print("Audit complete. Original input CSV was not modified.")
    print(f"JSON summary: {json_path}")
    print(f"Per-row score export: {csv_path}")
    print("No calibration was fitted and no threshold/verdict was changed.")


if __name__ == "__main__":
    main()
