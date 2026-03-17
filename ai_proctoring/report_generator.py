"""
Session Report Generator v3
Professional PDF exam session report with all violation types,
timeline, screenshots, risk score, and per-module summary.
"""

import os, json
from datetime import datetime
from collections import Counter

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

C_RED    = colors.HexColor("#C0392B")
C_ORANGE = colors.HexColor("#E67E22")
C_GREEN  = colors.HexColor("#27AE60")
C_BLUE   = colors.HexColor("#2980B9")
C_NAVY   = colors.HexColor("#1A252F")
C_DARK   = colors.HexColor("#2C3E50")
C_LIGHT  = colors.HexColor("#F4F6F7")
C_MID    = colors.HexColor("#BDC3C7")
C_WHITE  = colors.white

VIOLATION_LABELS = {
    "NO_FACE":          ("No Face Detected",        C_RED),
    "MULTIPLE_FACES":   ("Multiple Faces",           C_RED),
    "LOOKING_AWAY":     ("Looking Away",             C_ORANGE),
    "PHONE_DETECTED":   ("Mobile Phone Detected",    C_RED),
    "AUDIO_VIOLATION":  ("Suspicious Audio",         C_ORANGE),
    "BLINK_ANOMALY":    ("Abnormal Blink Rate",      C_ORANGE),
    "TAB_SWITCH":       ("Tab / Window Switch",      C_ORANGE),
    "FORBIDDEN_OBJECT": ("Forbidden Object",         C_RED),
    "TALKING_DETECTED": ("Talking / Whispering",     C_RED),
    "CAMERA_BLOCKED":   ("Camera Blocked",           C_RED),
    "LIGHTING_CHANGE":  ("Suspicious Lighting",      C_ORANGE),
    "BACKGROUND_MOTION":("Background Motion",        C_ORANGE),
    "IDENTITY_MISMATCH":("Identity Mismatch",        C_RED),
}


def generate_report(
    log_path:      str = "violations.jsonl",
    output_path:   str = "exam_report.pdf",
    session_start: str = None,
    candidate_id:  str = "CANDIDATE-001",
    exam_name:     str = "General Examination",
    institution:   str = "Institution",
    strictness:    str = "moderate",
) -> str:
    events        = _load_events(log_path)
    session_end   = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    session_start = session_start or session_end

    # Calculate duration
    try:
        from datetime import datetime as dt
        t1 = dt.strptime(session_start, "%Y-%m-%dT%H:%M:%S")
        t2 = dt.strptime(session_end,   "%Y-%m-%dT%H:%M:%S")
        duration = str(t2 - t1).split(".")[0]
    except Exception:
        duration = "N/A"

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.8*cm,  bottomMargin=1.8*cm,
    )
    styles = _build_styles()
    story  = []

    # ── Title block ───────────────────────────────────────────────────────────
    story.append(Paragraph(institution.upper(), styles["institution"]))
    story.append(Paragraph("AI EXAM PROCTORING REPORT", styles["main_title"]))
    story.append(HRFlowable(width="100%", thickness=3, color=C_NAVY))
    story.append(Spacer(1, 0.5*cm))

    # ── Session info grid ─────────────────────────────────────────────────────
    risk_level, risk_color = _risk_assessment(events)
    risk_score             = _risk_score(events)

    info_data = [
        ["Candidate ID",  candidate_id,   "Exam Name",   exam_name],
        ["Session Start", session_start,  "Session End", session_end],
        ["Duration",      duration,       "Strictness",  strictness.title()],
        ["Total Violations", str(len(events)), "Risk Score",
         Paragraph(f'<font color="#{_hex(risk_color)}"><b>{risk_score}/100 — {risk_level}</b></font>',
                   styles["body"])],
    ]
    info_table = Table(info_data, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,-1), C_LIGHT),
        ("BACKGROUND",    (2,0),(2,-1), C_LIGHT),
        ("FONTNAME",      (0,0),(-1,-1),"Helvetica"),
        ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTNAME",      (2,0),(2,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("GRID",          (0,0),(-1,-1), 0.5, C_MID),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [C_WHITE, C_LIGHT]),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.7*cm))

    # ── Violation summary ─────────────────────────────────────────────────────
    story.append(_section_header("Violation Summary", styles))
    if not events:
        story.append(Paragraph("✓  No violations recorded. Clean exam session.", styles["ok"]))
    else:
        counts = Counter(e["violation_type"] for e in events)
        rows   = [["Violation Type", "Count", "% of Total", "Avg Confidence", "Severity"]]
        for vtype, count in sorted(counts.items(), key=lambda x: -x[1]):
            vtype_evts = [e for e in events if e["violation_type"] == vtype]
            avg_conf   = sum(e["confidence"] for e in vtype_evts) / len(vtype_evts)
            pct        = f"{count / len(events) * 100:.1f}%"
            label, vc  = VIOLATION_LABELS.get(vtype, (vtype, C_DARK))
            severity   = "HIGH" if vc == C_RED else "MEDIUM"
            rows.append([label, str(count), pct, f"{avg_conf:.2f}", severity])

        vt = Table(rows, colWidths=[5.5*cm, 2*cm, 2.5*cm, 3*cm, 2.5*cm])
        vt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), C_NAVY),
            ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTNAME",      (0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",      (0,0),(-1,-1), 8.5),
            ("GRID",          (0,0),(-1,-1), 0.4, C_MID),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_LIGHT]),
            ("ALIGN",         (1,0),(-1,-1), "CENTER"),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ]))
        story.append(vt)

    story.append(Spacer(1, 0.7*cm))

    # ── Event timeline ────────────────────────────────────────────────────────
    story.append(_section_header("Event Timeline", styles))
    if events:
        rows = [["#", "Timestamp", "Violation Type", "Conf", "Details"]]
        for i, ev in enumerate(events[:60], 1):
            label, _ = VIOLATION_LABELS.get(ev["violation_type"], (ev["violation_type"], C_DARK))
            extra_str = ""
            if ev.get("extra"):
                ex = ev["extra"]
                if "direction" in ex:      extra_str = f"dir={ex['direction']}"
                elif "object" in ex:       extra_str = f"obj={ex['object']}"
                elif "reason" in ex:       extra_str = ex["reason"]
                elif "status" in ex:       extra_str = ex["status"]
                elif "to_window" in ex:    extra_str = ex.get("to_window","")[:20]
                elif "similarity" in ex:   extra_str = f"sim={ex['similarity']}"
                elif "motion_pct" in ex:   extra_str = f"{ex['motion_pct']}% area"
            rows.append([str(i), ev["timestamp"], label,
                         f"{ev['confidence']:.2f}", extra_str])
        tl = Table(rows, colWidths=[0.8*cm, 4*cm, 4.5*cm, 1.8*cm, 4.4*cm])
        tl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), C_BLUE),
            ("TEXTCOLOR",     (0,0),(-1,0), C_WHITE),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTNAME",      (0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",      (0,0),(-1,-1), 7.5),
            ("GRID",          (0,0),(-1,-1), 0.4, C_MID),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_LIGHT]),
            ("ALIGN",         (0,0),(0,-1), "CENTER"),
            ("ALIGN",         (3,0),(3,-1), "CENTER"),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ]))
        story.append(tl)
        if len(events) > 60:
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph(
                f"... and {len(events)-60} more events. See {C_LOG_FILE} for full log.",
                styles["note"]))

    story.append(Spacer(1, 0.7*cm))

    # ── Screenshots ───────────────────────────────────────────────────────────
    screenshots = [
        e["screenshot_path"] for e in events
        if os.path.exists(e.get("screenshot_path", ""))
    ][:9]   # Up to 9 (3×3 grid)

    if screenshots:
        story.append(_section_header("Violation Screenshots", styles))
        story.append(Spacer(1, 0.3*cm))
        THUMB = 5.2*cm
        rows, row = [], []
        for i, path in enumerate(screenshots):
            try:
                img = RLImage(path, width=THUMB, height=THUMB*0.75)
                row.append(img)
            except Exception:
                row.append(Paragraph("(error)", styles["note"]))
            if len(row) == 3:
                rows.append(row); row = []
        if row:
            while len(row) < 3: row.append(Paragraph("", styles["body"]))
            rows.append(row)
        if rows:
            img_tbl = Table(rows, colWidths=[THUMB+0.5*cm]*3)
            img_tbl.setStyle(TableStyle([
                ("ALIGN",         (0,0),(-1,-1), "CENTER"),
                ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
                ("GRID",          (0,0),(-1,-1), 0.4, C_MID),
                ("ROWBACKGROUNDS",(0,0),(-1,-1), [C_LIGHT]),
                ("TOPPADDING",    (0,0),(-1,-1), 5),
                ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ]))
            story.append(img_tbl)

    story.append(Spacer(1, 0.8*cm))

    # ── Risk assessment footer ────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=2, color=C_DARK))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        f"Risk Assessment: <font color='#{_hex(risk_color)}'><b>{risk_level} ({risk_score}/100)</b></font>",
        styles["risk_footer"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(_risk_note(risk_level, len(events)), styles["body"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        f"Generated by AI Proctoring System v3  •  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["footer"]))

    doc.build(story)
    print(f"[ReportGenerator] Report saved: '{output_path}'")
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

C_LOG_FILE = "violations.jsonl"

def _load_events(path):
    events = []
    if not os.path.exists(path): return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: events.append(json.loads(line))
                except: pass
    return events

def _risk_score(events) -> int:
    HIGH_WEIGHT = {"NO_FACE":10,"MULTIPLE_FACES":15,"PHONE_DETECTED":20,
                   "FORBIDDEN_OBJECT":15,"TALKING_DETECTED":12,
                   "CAMERA_BLOCKED":18,"IDENTITY_MISMATCH":20}
    MED_WEIGHT  = {"LOOKING_AWAY":5,"AUDIO_VIOLATION":5,"BLINK_ANOMALY":3,
                   "TAB_SWITCH":8,"LIGHTING_CHANGE":4,"BACKGROUND_MOTION":3}
    score = 0
    for e in events:
        vt = e["violation_type"]
        score += HIGH_WEIGHT.get(vt, MED_WEIGHT.get(vt, 2))
    return min(100, score)

def _risk_assessment(events):
    score = _risk_score(events)
    if score == 0:   return "CLEAN",  C_GREEN
    if score <= 20:  return "LOW",    C_GREEN
    if score <= 50:  return "MEDIUM", C_ORANGE
    return "HIGH", C_RED

def _risk_note(level, count):
    notes = {
        "CLEAN":  "No violations detected. Candidate behaviour appears completely normal.",
        "LOW":    f"{count} minor violation(s). Likely accidental. No action required.",
        "MEDIUM": f"{count} violation(s) detected. Suspicious behaviour observed. Manual review recommended.",
        "HIGH":   f"{count} violation(s) detected. HIGH probability of academic dishonesty. Immediate review required.",
    }
    return notes.get(level, "")

def _hex(color) -> str:
    r, g, b = int(color.red*255), int(color.green*255), int(color.blue*255)
    return f"{r:02X}{g:02X}{b:02X}"

def _section_header(title, styles):
    return KeepTogether([
        Paragraph(title, styles["section_title"]),
        HRFlowable(width="100%", thickness=1, color=C_MID),
        Spacer(1, 0.2*cm),
    ])

def _build_styles():
    base, S = getSampleStyleSheet(), {}
    S["institution"]  = ParagraphStyle("institution", parent=base["Normal"],
        fontSize=9, textColor=C_DARK, alignment=TA_CENTER, spaceAfter=2)
    S["main_title"]   = ParagraphStyle("main_title", parent=base["Title"],
        fontSize=20, textColor=C_NAVY, alignment=TA_CENTER, spaceAfter=8)
    S["section_title"]= ParagraphStyle("section_title", parent=base["Heading2"],
        fontSize=11, textColor=C_DARK, spaceBefore=4, spaceAfter=3)
    S["body"]         = ParagraphStyle("body", parent=base["Normal"],
        fontSize=8.5, leading=13)
    S["ok"]           = ParagraphStyle("ok", parent=base["Normal"],
        fontSize=9, textColor=C_GREEN, leading=14)
    S["note"]         = ParagraphStyle("note", parent=base["Normal"],
        fontSize=7.5, textColor=colors.grey, leading=11)
    S["footer"]       = ParagraphStyle("footer", parent=base["Normal"],
        fontSize=7.5, textColor=colors.grey, alignment=TA_CENTER)
    S["risk_footer"]  = ParagraphStyle("risk_footer", parent=base["Normal"],
        fontSize=13, fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4)
    return S
