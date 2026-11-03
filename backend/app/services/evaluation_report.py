from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any


from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

NAVY = colors.HexColor("#14243A")
BLUE = colors.HexColor("#1676A8")
LIGHT_BLUE = colors.HexColor("#EAF3F9")
GREY = colors.HexColor("#64748B")
LIGHT_GREY = colors.HexColor("#F1F5F9")
AMBER = colors.HexColor("#FFF4D6")


def _styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=23,
            leading=28,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontSize=13,
            leading=17,
            textColor=BLUE,
            spaceBefore=12,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=GREY,
        )
    )

    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["Normal"],
            fontSize=9,
            leading=14,
        )
    )

    return styles


def _safe(value: Any) -> str:
    if value is None:
        return "Not available"
    return str(value)


def _percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{value * 100:.2f}%"


def _table(data, widths=None):
    table = Table(data, colWidths=widths, repeatRows=1)

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    return table


def _metric_rows(metrics):
    fields = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision_manipulated_class_1"),
        ("Recall", "recall_manipulated_class_1"),
        ("F1 Score", "f1_manipulated_class_1"),
        ("ROC-AUC", "roc_auc"),
        ("Average Precision", "average_precision"),
    ]

    rows = [["Metric", "Value"]]

    for label, key in fields:
        rows.append([label, _percent(metrics.get(key))])

    return rows


def _confusion_rows(metrics):
    matrix = metrics.get("confusion_matrix_labels_0_original_1_manipulated") or {}

    return [
        ["Actual / Predicted", "Original", "Manipulated"],
        ["Original", _safe(matrix.get("tn")), _safe(matrix.get("fp"))],
        ["Manipulated", _safe(matrix.get("fn")), _safe(matrix.get("tp"))],
    ]


def _header_footer(canvas, document):
    canvas.saveState()

    width, height = A4

    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 9 * mm, width, 9 * mm, fill=1, stroke=0)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)

    canvas.drawString(
        18 * mm,
        12 * mm,
        "VERIFRAME | EVIDENCE-FIRST VIDEO FORENSICS",
    )

    canvas.drawRightString(
        width - 18 * mm,
        12 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


def build_evaluation_pdf(job: dict[str, Any]) -> bytes:
    result = job.get("result") or {}
    summary = result.get("summary") or {}

    if job.get("status") != "completed" or not summary:
        raise ValueError("A completed evaluation with summary metrics is required.")

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=20 * mm,
    )

    styles = _styles()
    story = []

    frame = summary.get("frame_metrics") or {}
    video = summary.get("video_metrics_mean_score_per_frame_folder") or {}

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("MODEL EVALUATION REPORT", styles["ReportTitle"]))
    story.append(
        Paragraph(
            "Custom DeepfakeBench Xception",
            ParagraphStyle(
                "Subtitle",
                parent=styles["Normal"],
                alignment=TA_CENTER,
                textColor=GREY,
                fontSize=10,
            ),
        )
    )
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("1. Evaluation details", styles["Section"]))

    details = [
        ["Field", "Value"],
        ["Evaluation ID", _safe(job.get("evaluation_id"))],
        ["Status", _safe(job.get("status"))],
        ["Dataset split", _safe(summary.get("split_evaluated"))],
        ["Created at", _safe(job.get("created_at"))],
        ["Completed at", _safe(job.get("completed_at"))],
        ["Model", _safe(summary.get("model"))],
        ["Input size", _safe(summary.get("input_size"))],
        ["Device", _safe(summary.get("device"))],
        ["GPU", _safe(summary.get("gpu_name"))],
        ["Threshold", _safe(summary.get("threshold"))],
        ["Checkpoint SHA-256", _safe(summary.get("checkpoint_sha256"))],
    ]

    story.append(_table(details, [48 * mm, 126 * mm]))

    story.append(Paragraph("2. Frame-level performance", styles["Section"]))
    story.append(
        Paragraph(
            f"Evaluated samples: {_safe(frame.get('n'))} frames",
            styles["BodyCustom"],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(_table(_metric_rows(frame), [85 * mm, 89 * mm]))

    story.append(Paragraph("3. Frame-level confusion matrix", styles["Section"]))
    story.append(
        _table(
            _confusion_rows(frame),
            [58 * mm, 58 * mm, 58 * mm],
        )
    )

    story.append(
        Paragraph(
            "4. Video-folder-level performance",
            styles["Section"],
        )
    )
    story.append(
        Paragraph(
            "Video-level results are calculated by aggregating frame scores "
            "within frame folders. They should not automatically be interpreted "
            "as independently verified source-video results.",
            styles["BodyCustom"],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            f"Evaluated aggregates: {_safe(video.get('n'))}",
            styles["BodyCustom"],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(_table(_metric_rows(video), [85 * mm, 89 * mm]))

    story.append(
        Paragraph(
            "5. Video-folder-level confusion matrix",
            styles["Section"],
        )
    )
    story.append(
        _table(
            _confusion_rows(video),
            [58 * mm, 58 * mm, 58 * mm],
        )
    )

    story.append(Paragraph("6. Dataset and evaluation notes", styles["Section"]))

    counts = summary.get("split_counts_in_csv") or {}

    split_data = [["Dataset split", "Frame count"]]
    for split_name, count in counts.items():
        split_data.append([str(split_name), str(count)])

    if len(split_data) > 1:
        story.append(_table(split_data, [85 * mm, 89 * mm]))
        story.append(Spacer(1, 4 * mm))

    warnings = summary.get("warnings") or []
    leakage = summary.get("multi_split_frame_folders") or []

    if warnings:
        story.append(Paragraph("Evaluation warnings:", styles["BodyCustom"]))
        for warning in warnings:
            story.append(
                Paragraph(
                    f"• {_safe(warning)}",
                    styles["SmallNote"],
                )
            )

    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            f"Frame-folder split overlap entries: {len(leakage)}",
            styles["SmallNote"],
        )
    )

    story.append(
        Paragraph("7. Research interpretation and limitations", styles["Section"])
    )

    story.append(
        Paragraph(
            "These metrics describe performance on the evaluated dataset split. "
            "They do not independently establish generalization to unseen "
            "datasets, identities, manipulation methods, or real-world videos. "
            "Video-folder metrics are based on aggregation across frame folders. "
            "ROC-AUC and Average Precision measure ranking performance and do "
            "not establish score calibration.",
            styles["BodyCustom"],
        )
    )

    story.append(Spacer(1, 4 * mm))

    story.append(
        Paragraph(
            "This report is generated by VeriFrame, a research prototype. "
            "It is not a certified forensic opinion or proof of authenticity "
            "or manipulation.",
            styles["SmallNote"],
        )
    )

    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph(
            "Report generated: " + datetime.now(timezone.utc).isoformat(),
            styles["SmallNote"],
        )
    )

    document.build(
        story,
        onFirstPage=_header_footer,
        onLaterPages=_header_footer,
    )

    return buffer.getvalue()


def build_comparison_pdf(comparison: dict[str, Any]) -> bytes:
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=20 * mm,
    )

    styles = _styles()
    story = []

    story.append(Spacer(1, 8 * mm))
    story.append(
        Paragraph(
            "MODEL EVALUATION COMPARISON",
            styles["ReportTitle"],
        )
    )
    story.append(
        Paragraph(
            "VeriFrame | Custom DeepfakeBench Xception",
            ParagraphStyle(
                "CompareSubtitle",
                parent=styles["Normal"],
                alignment=TA_CENTER,
                textColor=GREY,
                fontSize=10,
            ),
        )
    )

    first = comparison.get("first") or {}
    second = comparison.get("second") or {}

    story.append(Paragraph("1. Compared evaluation runs", styles["Section"]))

    run_rows = [
        ["Field", "First evaluation", "Second evaluation"],
        [
            "Evaluation ID",
            _safe(first.get("evaluation_id")),
            _safe(second.get("evaluation_id")),
        ],
        [
            "Created at",
            _safe(first.get("created_at")),
            _safe(second.get("created_at")),
        ],
        [
            "Dataset split",
            _safe(first.get("split")),
            _safe(second.get("split")),
        ],
        [
            "Checkpoint SHA-256",
            Paragraph(
                _safe(first.get("checkpoint_sha256")),
                ParagraphStyle(
                    "HashFirst",
                    fontName="Helvetica",
                    fontSize=6,
                    leading=8,
                    wordWrap="CJK",
                ),
            ),
            Paragraph(
                _safe(second.get("checkpoint_sha256")),
                ParagraphStyle(
                    "HashSecond",
                    fontName="Helvetica",
                    fontSize=6,
                    leading=8,
                    wordWrap="CJK",
                ),
            ),
        ],
    ]

    story.append(_table(run_rows, [36 * mm, 75 * mm, 75 * mm]))

    story.append(Paragraph("2. Comparison summary", styles["Section"]))

    story.append(
        Paragraph(
            f"Same dataset split: {comparison.get('same_split')}",
            styles["BodyCustom"],
        )
    )
    story.append(
        Paragraph(
            f"Same checkpoint: {comparison.get('same_checkpoint')}",
            styles["BodyCustom"],
        )
    )
    story.append(
        Paragraph(
            _safe(comparison.get("delta_note")),
            styles["SmallNote"],
        )
    )

    metric_labels = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
        ("average_precision", "Average Precision"),
    ]

    def comparison_table(level):
        values = comparison.get(level) or {}
        rows = [["Metric", "First", "Second", "Difference (pp)"]]

        for key, label in metric_labels:
            metric = values.get(key) or {}
            delta = metric.get("delta")

            rows.append(
                [
                    label,
                    _percent(metric.get("first")),
                    _percent(metric.get("second")),
                    (
                        f"{delta * 100:+.2f}"
                        if isinstance(delta, (int, float))
                        else "N/A"
                    ),
                ]
            )

        return rows

    story.append(Paragraph("3. Frame-level comparison", styles["Section"]))
    story.append(
        _table(
            comparison_table("frame_level"),
            [49 * mm, 39 * mm, 39 * mm, 59 * mm],
        )
    )

    story.append(
        Paragraph(
            "4. Video-folder-level comparison",
            styles["Section"],
        )
    )
    story.append(
        _table(
            comparison_table("video_level"),
            [49 * mm, 39 * mm, 39 * mm, 59 * mm],
        )
    )

    story.append(Paragraph("5. Comparison warnings", styles["Section"]))

    warnings = comparison.get("warnings") or []

    if warnings:
        for warning in warnings:
            story.append(
                Paragraph(
                    f"• {_safe(warning)}",
                    styles["BodyCustom"],
                )
            )
    else:
        story.append(
            Paragraph(
                "No comparison warnings were returned by the API.",
                styles["BodyCustom"],
            )
        )

    story.append(
        Paragraph(
            "6. Interpretation and limitations",
            styles["Section"],
        )
    )

    story.append(
        Paragraph(
            "Metric differences are calculated as second evaluation minus "
            "first evaluation. Differences should only be attributed to model "
            "changes when the dataset, split, preprocessing, threshold, and "
            "evaluation protocol are controlled. Identical checkpoint hashes "
            "mean that the model weights are identical.",
            styles["BodyCustom"],
        )
    )

    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph(
            "This comparison is a research evaluation artifact, not a "
            "certified forensic opinion. The underlying detector scores "
            "are not necessarily calibrated probabilities.",
            styles["SmallNote"],
        )
    )

    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph(
            "Report generated: " + datetime.now(timezone.utc).isoformat(),
            styles["SmallNote"],
        )
    )

    document.build(
        story,
        onFirstPage=_header_footer,
        onLaterPages=_header_footer,
    )

    return buffer.getvalue()
