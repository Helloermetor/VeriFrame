from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Flowable,
)


def _safe(value, fallback="Not available"):
    if value is None or value == "":
        return fallback
    if isinstance(value, (dict, list)):
        value = str(value)
    return escape(str(value))


def _paragraph(text, style):
    return Paragraph(text, style)



class CoverPage(Flowable):
    """A clean, single-page report cover drawn with ReportLab primitives."""
    def __init__(self, filename, analysis_id, outcome, created_at, generated_at):
        super().__init__()
        self.filename = str(filename or "Untitled video")
        self.analysis_id = str(analysis_id or "Not available")
        self.outcome = str(outcome or "INCONCLUSIVE")
        self.created_at = str(created_at or "Not recorded")
        self.generated_at = str(generated_at)
        self.width = A4[0] - 36 * mm
        self.height = A4[1] - 36 * mm

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        self.height = availHeight
        return availWidth, availHeight

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        navy = colors.HexColor("#14243A")
        blue = colors.HexColor("#2563EB")
        muted = colors.HexColor("#64748B")
        c.setFillColor(navy)
        c.roundRect(0, h - 105 * mm, w, 105 * mm, 5 * mm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#93C5FD"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(14 * mm, h - 20 * mm, "VERIFRAME  /  VIDEO FORENSICS")
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 25)
        c.drawString(14 * mm, h - 48 * mm, "Forensic Analysis")
        c.drawString(14 * mm, h - 60 * mm, "Report")
        c.setFillColor(colors.HexColor("#CBD5E1"))
        c.setFont("Helvetica", 9)
        c.drawString(14 * mm, h - 76 * mm, "Evidence summary · Model outputs · Stability diagnostics")

        c.setFillColor(muted)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(2 * mm, h - 125 * mm, "VIDEO FILE")
        c.setFillColor(navy)
        c.setFont("Helvetica-Bold", 14)
        name = self.filename
        if len(name) > 62:
            name = name[:59] + "..."
        c.drawString(2 * mm, h - 135 * mm, name)

        c.setFillColor(muted)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(2 * mm, h - 158 * mm, "CURRENT ASSESSMENT")
        c.setFillColor(colors.HexColor("#FFF7E6"))
        c.roundRect(0, h - 181 * mm, w, 15 * mm, 3 * mm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#8A5A00"))
        c.setFont("Helvetica-Bold", 12)
        c.drawString(6 * mm, h - 176 * mm, self.outcome.upper())

        c.setFillColor(muted)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(2 * mm, h - 203 * mm, "ANALYSIS ID")
        c.setFillColor(navy)
        c.setFont("Helvetica", 9)
        c.drawString(2 * mm, h - 210 * mm, self.analysis_id)
        c.setFillColor(muted)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(2 * mm, h - 228 * mm, "ANALYSIS CREATED")
        c.setFillColor(navy)
        c.setFont("Helvetica", 9)
        c.drawString(2 * mm, h - 235 * mm, self.created_at)

        c.setStrokeColor(colors.HexColor("#CBD5E1"))
        c.line(0, 18 * mm, w, 18 * mm)
        c.setFillColor(muted)
        c.setFont("Helvetica", 8)
        c.drawString(0, 11 * mm, "Generated " + self.generated_at)
        c.drawRightString(w, 11 * mm, "RESEARCH PROTOTYPE · NOT A CERTIFICATION")


class ScoreChart(Flowable):
    """Compact bar chart for frame-level uncalibrated model scores."""
    def __init__(self, values, title, labels=None, x_axis_label="Sampled frames (up to first 60 shown)", width=174*mm, height=58*mm, y_max=1.0):
        super().__init__()
        self.values = values[:60]
        self.labels = (labels or [])[:60]
        self.x_axis_label = x_axis_label
        self.title = title
        self.width = width
        self.height = height
        self.y_max = y_max

    def wrap(self, availWidth, availHeight):
        self.width = min(self.width, availWidth)
        return self.width, self.height

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        left, right, bottom, top = 12*mm, 3*mm, 12*mm, 10*mm
        plot_w, plot_h = w-left-right, h-bottom-top
        c.setFillColor(colors.HexColor("#17365D"))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(0, h-4*mm, self.title)
        c.setStrokeColor(colors.HexColor("#CBD5E1"))
        c.line(left, bottom, left, bottom+plot_h)
        c.line(left, bottom, left+plot_w, bottom)
        for fraction in (0, .25, .5, .75, 1):
            y = bottom + plot_h*fraction
            c.setStrokeColor(colors.HexColor("#E2E8F0"))
            c.line(left, y, left+plot_w, y)
            c.setFillColor(colors.HexColor("#64748B"))
            c.setFont("Helvetica", 6)
            c.drawRightString(left-2*mm, y-2, f"{self.y_max*fraction:.2f}")
        if not self.values:
            c.setFillColor(colors.HexColor("#64748B"))
            c.setFont("Helvetica", 8)
            c.drawString(left+4*mm, bottom+plot_h/2, "No numeric frame scores available")
            return
        slot = plot_w / len(self.values)
        bar_w = max(1, slot*.62)
        for i, value in enumerate(self.values):
            v = max(0, min(self.y_max, value))
            bh = plot_h * v / self.y_max if self.y_max else 0
            x = left + i*slot + (slot-bar_w)/2
            c.setFillColor(colors.HexColor("#3B82F6"))
            c.rect(x, bottom, bar_w, bh, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#64748B"))
        c.setFont("Helvetica", 6)
        c.drawString(left, 2*mm, self.x_axis_label)


def _format_value(value):
    if value is None or value == "":
        return "Not available"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _first_value(data, *keys):
    for key in keys:
        value = data.get(key) if isinstance(data, dict) else None
        if value is not None and value != "":
            return value
    return None


def _numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_analysis_pdf(record: dict) -> bytes:
    """Build a detector-complete PDF from the stored VeriFrame analysis record."""
    import json

    result = record.get("result") or {}
    metadata = result.get("metadata") or {}
    evidence = result.get("evidence") or {}
    committee = result.get("committee") or {}
    stability = result.get("stability") or {}
    outcome = result.get("outcome") or "INCONCLUSIVE"
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    # Detector names are explicit and ordered. Unknown committee entries are
    # retained after these so a newly integrated detector is not silently lost.
    detector_names = [
        ("gend", "GenD CLIP ViT-L/14"),
        ("xception", "Custom DeepfakeBench Xception"),
        ("effort", "Effort (CLIP ViT-L/14 + SVD)"),
        ("fakestormer", "FakeSTormer (temporal localization)"),
    ]
    detectors = []
    seen = set()
    for key, label in detector_names:
        data = committee.get(key)
        if isinstance(data, dict) and data:
            detectors.append((key, label, data))
            seen.add(key)
    if isinstance(committee, dict):
        for key, data in committee.items():
            if key not in seen and isinstance(data, dict) and data:
                detectors.append((key, str(key).replace("_", " ").title(), data))

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=f"VeriFrame Forensic Report - {record.get('filename', 'Video')}",
        author="VeriFrame",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="VFTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=colors.HexColor("#15233B"), alignment=TA_CENTER, spaceAfter=5 * mm))
    styles.add(ParagraphStyle(name="VFSubtitle", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.HexColor("#64748B"), fontSize=9, spaceAfter=8 * mm))
    styles.add(ParagraphStyle(name="VFSection", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#17365D"), spaceBefore=5 * mm, spaceAfter=2 * mm))
    styles.add(ParagraphStyle(name="VFSubsection", parent=styles["Heading3"], fontSize=10, leading=13, textColor=colors.HexColor("#1E3A5F"), spaceBefore=3 * mm, spaceAfter=1.5 * mm))
    styles.add(ParagraphStyle(name="VFBody", parent=styles["BodyText"], fontSize=8.5, leading=12, spaceAfter=2 * mm))
    styles.add(ParagraphStyle(name="VFSmall", parent=styles["BodyText"], fontSize=7, leading=9, textColor=colors.HexColor("#475569")))

    created = record.get("created_at") or "Not recorded"
    try:
        created = datetime.fromisoformat(str(created).replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        created = str(created)

    story = [
        CoverPage(record.get("filename"), record.get("analysis_id"), outcome, created, generated),
        PageBreak(),
        _paragraph("VERIFRAME · DETAILED FINDINGS", styles["VFTitle"]),
        _paragraph("VIDEO FORENSIC ANALYSIS REPORT", styles["VFSubtitle"]),
    ]
    story.append(_paragraph("CURRENT ASSESSMENT", styles["VFSection"]))
    note = result.get("outcome_note") or "The stored analysis does not contain an outcome explanation."
    verdict = Table([[_paragraph(f"<b>{_safe(outcome)}</b>", styles["VFBody"])], [_paragraph(_safe(note), styles["VFBody"])]], colWidths=[174 * mm])
    verdict.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7E6")), ("BOX", (0, 0), (-1, -1), .7, colors.HexColor("#E7B95B")), ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(verdict)
    story.append(_paragraph("ANALYSIS IDENTIFICATION", styles["VFSection"]))
    identification = [
        ["Filename", _safe(record.get("filename"))], ["Analysis ID", _safe(record.get("analysis_id"))],
        ["Created at", _safe(created)], ["Status", _safe(record.get("status"))],
        ["File size", f"{record.get('size_bytes', 'Not available')} bytes"],
        ["Original SHA-256", _safe(record.get("sha256"))],
    ]
    table = Table(identification, colWidths=[42 * mm, 132 * mm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F7")), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CBD5E1")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(table)

    story.append(_paragraph("VIDEO METADATA", styles["VFSection"]))
    metadata_rows = [
        ["Container", _safe(metadata.get("container"))], ["Video codec", _safe(metadata.get("video_codec"))],
        ["Dimensions", f"{metadata.get('width')} × {metadata.get('height')}" if metadata.get("width") and metadata.get("height") else "Not available"],
        ["Duration", f"{float(metadata['duration_seconds']):.2f} seconds" if isinstance(metadata.get("duration_seconds"), (int, float)) else "Not available"],
        ["Source FPS", _safe(metadata.get("source_fps"))], ["Sampled frames", _safe(metadata.get("sampled_frame_count"))],
    ]
    mt = Table(metadata_rows, colWidths=[42 * mm, 132 * mm])
    mt.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F7")), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CBD5E1")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(mt)

    story.append(_paragraph("DETECTOR SUMMARY", styles["VFSection"]))
    summary_rows = [["Detector", "Status", "Primary score / output", "Checkpoint"]]
    for key, label, data in detectors:
        status = _first_value(data, "status", "inference_status", "state")
        score = _first_value(data, "class1_manipulated_score_mean", "manipulated_score", "class1_manipulated_score", "temporal_localization_final", "final_score", "score")
        checkpoint = _first_value(data, "checkpoint_filename", "checkpoint", "weights", "model_checkpoint")
        summary_rows.append([_safe(label), _safe(status or "completed"), _safe(_format_value(score)), _safe(checkpoint)])
    if len(summary_rows) == 1:
        summary_rows.append(["No detector records found", "Not available", "Not available", "Not available"])
    st = Table(summary_rows, colWidths=[48 * mm, 30 * mm, 48 * mm, 48 * mm], repeatRows=1)
    st.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#CBD5E1")), ("FONTSIZE", (0, 0), (-1, -1), 6.5), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(st)

    # Full per-detector scalar metadata and scores; nested arrays are rendered
    # in dedicated frame/temporal tables below to avoid dumping raw JSON blobs.
    for key, label, data in detectors:
        story.append(_paragraph(label.upper(), styles["VFSubsection"]))
        rows = []
        for field, value in data.items():
            if isinstance(value, (dict, list, tuple)) or value is None:
                continue
            rows.append([_safe(str(field).replace("_", " ").title()), _safe(_format_value(value))])
        if rows:
            t = Table(rows, colWidths=[58 * mm, 116 * mm])
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F7")), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#CBD5E1")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            story.append(t)
        else:
            story.append(_paragraph("No scalar detector metadata was stored.", styles["VFSmall"]))

    # Frame-level data is kept detector-specific. The top-level frames array
    # remains the Xception-compatible legacy source if no detector frame list exists.
    story.append(_paragraph("FRAME-LEVEL EVIDENCE", styles["VFSection"]))
    frame_sources = []
    for key, label, data in detectors:
        detector_frames = data.get("frames") if isinstance(data.get("frames"), list) else None
        if detector_frames:
            frame_sources.append((label, detector_frames))
    xception_data = committee.get("xception") if isinstance(committee, dict) else None
    xception_has_frames = isinstance(xception_data, dict) and bool(xception_data.get("frames"))
    if not xception_has_frames and isinstance(result.get("frames"), list) and result.get("frames"):
        frame_sources.append(("Custom DeepfakeBench Xception (stored result.frames)", result["frames"]))
    if frame_sources:
        for label, detector_frames in frame_sources:
            story.append(_paragraph(_safe(label), styles["VFSubsection"]))
            rows = [["Frame index", "Timestamp (s)", "Uncalibrated manipulated score", "Threshold output"]]
            for frame in detector_frames[:100]:
                if not isinstance(frame, dict):
                    continue
                score = _first_value(frame, "class1_manipulated_score", "manipulated_score", "score")
                rows.append([_safe(_first_value(frame, "frame_index", "index")), _safe(_first_value(frame, "timestamp_seconds", "timestamp")), _safe(_format_value(score)), _safe(_first_value(frame, "threshold_model_output", "threshold_output", "prediction"))])
            ft = Table(rows, colWidths=[27 * mm, 30 * mm, 78 * mm, 37 * mm], repeatRows=1)
            ft.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#CBD5E1")), ("FONTSIZE", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story.append(ft)
            if len(detector_frames) > 100:
                story.append(_paragraph(f"Showing the first 100 of {len(detector_frames)} stored frame records for {label}.", styles["VFSmall"]))
    else:
        story.append(_paragraph("No detector-specific frame records were stored for this analysis.", styles["VFBody"]))

    # FakeSTormer is temporal evidence, not a standard per-frame classifier score.
    fake_data = committee.get("fakestormer") if isinstance(committee, dict) else None
    if isinstance(fake_data, dict) and fake_data:
        story.append(_paragraph("FAKESTORMER TEMPORAL LOCALIZATION", styles["VFSection"]))
        temporal_values = _first_value(fake_data, "temporal_localization_values", "temporal_scores", "temporal_samples") or []
        sampled_indices = _first_value(fake_data, "sampled_frame_indices", "frame_indices") or []
        temporal_rows = [["Sample", "Frame index", "Timestamp (s)", "Temporal score"]]
        if isinstance(temporal_values, list):
            for i, item in enumerate(temporal_values):
                if isinstance(item, dict):
                    idx = _first_value(item, "frame_index", "index")
                    ts = _first_value(item, "timestamp_seconds", "timestamp")
                    val = _first_value(item, "score", "value", "temporal_localization_value")
                else:
                    idx = sampled_indices[i] if isinstance(sampled_indices, list) and i < len(sampled_indices) else None
                    ts = None
                    val = item
                temporal_rows.append([str(i + 1), _safe(idx), _safe(ts), _safe(_format_value(val))])
        if len(temporal_rows) > 1:
            tt = Table(temporal_rows, colWidths=[25 * mm, 40 * mm, 45 * mm, 62 * mm], repeatRows=1)
            tt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#CBD5E1")), ("FONTSIZE", (0, 0), (-1, -1), 7), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            story.append(tt)
        else:
            story.append(_paragraph("No temporal sample values were stored.", styles["VFBody"]))
        temporal_summary = [
            ["Final temporal-localization value", _safe(_format_value(_first_value(fake_data, "temporal_localization_final", "final_temporal_score", "temporal_score")))],
            ["Threshold", _safe(_format_value(_first_value(fake_data, "threshold", "decision_threshold")))],
            ["Official model label", _safe(_first_value(fake_data, "official_label", "predicted_label", "label"))],
            ["Sampled frame indices", _safe(_format_value(sampled_indices))],
        ]
        ts_table = Table(temporal_summary, colWidths=[58 * mm, 116 * mm])
        ts_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F7")), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#CBD5E1")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story.append(ts_table)
        story.append(_paragraph("FakeSTormer temporal-localization values are reported separately and are not treated as ordinary frame-classification scores.", styles["VFSmall"]))

    story.append(_paragraph("SCORE INTERPRETATION", styles["VFSection"]))
    story.append(_paragraph("Detector scores are uncalibrated model outputs. They are not validated probabilities of authenticity or manipulation. Outputs from different detectors may use different heads, thresholds, sampling procedures and score semantics; they should not be directly compared as if they shared a common scale. The overall VeriFrame assessment remains INCONCLUSIVE unless the stored analysis explicitly reports otherwise.", styles["VFBody"]))

    story.append(_paragraph("STABILITY TESTS", styles["VFSection"]))
    perturbations = stability.get("perturbations") or []
    if stability.get("status") in {"completed", "partial"} or perturbations:
        story.append(_paragraph(f"Status: {_safe(stability.get('status'))}; baseline score: {_safe(stability.get('baseline_score'))}; mean absolute score change: {_safe(_first_value(stability, 'mean_absolute_score_change', 'score'))}; decision flips: {_safe(stability.get('decision_flip_count'))}.", styles["VFBody"]))
        if perturbations:
            rows = [["Test", "Status", "Transformed score", "Absolute change", "Decision changed"]]
            for item in perturbations:
                rows.append([_safe(item.get("label") or item.get("id")), _safe(item.get("status")), _safe(_format_value(item.get("score"))), _safe(_format_value(item.get("absolute_score_delta"))), _safe(_format_value(item.get("decision_changed")))])
            pt = Table(rows, colWidths=[48 * mm, 25 * mm, 32 * mm, 34 * mm, 35 * mm], repeatRows=1)
            pt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#CBD5E1")), ("FONTSIZE", (0, 0), (-1, -1), 6.5), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            story.append(pt)
    else:
        story.append(_paragraph("Stability testing has not been run for this analysis.", styles["VFBody"]))

    story.append(PageBreak())
    story.append(_paragraph("LIMITATIONS AND CONTRADICTORY EVIDENCE", styles["VFSection"]))
    raw_limitations = evidence.get("limitations") or result.get("limitations") or []
    raw_contradictory = evidence.get("contradictory_evidence") or []
    source_items = [str(item).strip() for item in [*raw_contradictory, *raw_limitations] if str(item).strip()]
    limitations = []
    if not detectors:
        limitations.append("No detector result records were found in the stored committee data.")
    if any("c2pa" in item.lower() or "content credentials" in item.lower() for item in source_items):
        limitations.append("C2PA manifest / Content Credentials validation was not performed for this analysis.")
    if any("grad-cam" in item.lower() or "grad cam" in item.lower() for item in source_items):
        limitations.append("Grad-CAM explanations and their faithfulness have not been validated or implemented.")
    if any("face-quality" in item.lower() or "face quality" in item.lower() or "face tracking" in item.lower() for item in source_items):
        limitations.append("Face-quality gating and robust face detection/tracking have not been fully validated.")
    if any("faceforensics" in item.lower() or "ai-generated" in item.lower() for item in source_items):
        limitations.append("The available training/evaluation domain does not establish generalization to all manipulation methods or fully AI-generated video.")
    if any("conformal" in item.lower() for item in source_items):
        limitations.append("Conformal prediction has not been implemented.")
    if perturbations:
        limitations.append("Diagnostic perturbation checks do not constitute comprehensive robustness validation.")
    if fake_data and not _first_value(fake_data, "temporal_localization_values", "temporal_scores", "temporal_samples"):
        limitations.append("FakeSTormer output is present, but temporal sample values were not stored in this report record.")
    limitations.extend(source_items)
    deduped = []
    seen_text = set()
    for item in limitations:
        normalized = " ".join(str(item).split()).lower()
        if normalized and normalized not in seen_text:
            seen_text.add(normalized)
            deduped.append(str(item))
    if deduped:
        for item in deduped:
            story.append(_paragraph("• " + _safe(item), styles["VFBody"]))
    else:
        story.append(_paragraph("No additional limitation details were stored in this analysis record.", styles["VFBody"]))

    story.append(_paragraph("PROVENANCE", styles["VFSection"]))
    provenance = evidence.get("provenance_evidence") or []
    if provenance:
        for item in provenance:
            if isinstance(item, dict):
                story.append(_paragraph(f"<b>{_safe(item.get('label'))}:</b> {_safe(item.get('value'))}", styles["VFBody"]))
            else:
                story.append(_paragraph(_safe(item), styles["VFBody"]))
    else:
        story.append(_paragraph(f"Original upload SHA-256: {_safe(record.get('sha256'))}", styles["VFBody"]))
    story.append(_paragraph(f"C2PA / Content Credentials status: {_safe(_first_value(evidence, 'c2pa_status', 'content_credentials_status') or result.get('c2pa_status') or 'Not checked')}", styles["VFBody"]))
    story.append(Spacer(1, 5 * mm))
    story.append(_paragraph(f"Report generated: {escape(generated)}. This report reflects the data stored by VeriFrame at generation time. It is a research evidence summary, not a certification of authenticity, a legal conclusion, or proof that the media is genuine or manipulated.", styles["VFSmall"]))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(18 * mm, 8 * mm, "VERIFRAME · RESEARCH PROTOTYPE")
        canvas.drawRightString(A4[0] - 18 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
