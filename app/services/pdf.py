"""PDF report generation for completed analyses.

Renders a one- or two-page PDF containing the classification map, summary
metrics, advice, class distribution, and the farmer's observation. Localized
for EN / FR / AR. Arabic strings are shaped + bidi-reordered before being
drawn so ReportLab (which has no built-in shaping engine) renders them
correctly.
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..core.config import get_settings
from ..models import Analysis

_FONTS_REGISTERED = False
LATIN_FONT = "AgroLatin"
ARABIC_FONT = "AgroArabic"


def _ensure_fonts() -> None:
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    fonts_dir = get_settings().static_dir / "fonts"
    pdfmetrics.registerFont(TTFont(LATIN_FONT, str(fonts_dir / "NotoSans-Regular.ttf")))
    pdfmetrics.registerFont(TTFont(ARABIC_FONT, str(fonts_dir / "Amiri-Regular.ttf")))
    _FONTS_REGISTERED = True


def _shape_arabic(text: str) -> str:
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(text))


_STRINGS = {
    "en": {
        "title": "Crop Analysis Report",
        "generated": "Generated",
        "source": "Source",
        "analysis_id": "Analysis ID",
        "predicted_crop": "Predicted crop",
        "health": "Health",
        "trajectory": "Trajectory",
        "confidence": "Mean confidence",
        "ndvi": "Mean NDVI",
        "ndre": "Mean NDRE",
        "savi": "Mean SAVI",
        "temporal_trend": "Temporal trend",
        "advice": "Agronomic advice",
        "class_distribution": "Class distribution",
        "class_label": "Class",
        "share": "Share",
        "classification_map": "Classification map",
        "farmer_observation": "Farmer's observation",
        "observed_on": "Observed on",
        "no_observation": "No observation recorded.",
    },
    "fr": {
        "title": "Rapport d'Analyse des Cultures",
        "generated": "Généré le",
        "source": "Source",
        "analysis_id": "ID de l'analyse",
        "predicted_crop": "Culture prédite",
        "health": "Santé",
        "trajectory": "Trajectoire",
        "confidence": "Confiance moyenne",
        "ndvi": "NDVI moyen",
        "ndre": "NDRE moyen",
        "savi": "SAVI moyen",
        "temporal_trend": "Tendance temporelle",
        "advice": "Conseil agronomique",
        "class_distribution": "Distribution des classes",
        "class_label": "Classe",
        "share": "Part",
        "classification_map": "Carte de classification",
        "farmer_observation": "Observation du producteur",
        "observed_on": "Observé le",
        "no_observation": "Aucune observation enregistrée.",
    },
    "ar": {
        "title": "تقرير تحليل المحاصيل",
        "generated": "تم إنشاؤه في",
        "source": "المصدر",
        "analysis_id": "معرّف التحليل",
        "predicted_crop": "المحصول المتوقع",
        "health": "الصحة",
        "trajectory": "المسار",
        "confidence": "متوسط الثقة",
        "ndvi": "متوسط NDVI",
        "ndre": "متوسط NDRE",
        "savi": "متوسط SAVI",
        "temporal_trend": "الاتجاه الزمني",
        "advice": "نصيحة زراعية",
        "class_distribution": "توزيع الفئات",
        "class_label": "الفئة",
        "share": "الحصة",
        "classification_map": "خريطة التصنيف",
        "farmer_observation": "ملاحظة المزارع",
        "observed_on": "تاريخ الملاحظة",
        "no_observation": "لا توجد ملاحظات مسجلة.",
    },
}


# Localized labels for the canonical health_status / trajectory enum codes.
# Mirrors `result.health_codes` and `result.trajectory_codes` in the frontend
# i18n bundles. Legacy free-text values fall through to the raw string.
_CODE_LABELS: dict[str, dict[str, dict[str, str]]] = {
    "en": {
        "health": {
            "CRITICAL": "Critical",
            "STRESSED": "Stressed",
            "MODERATE": "Moderate",
            "HEALTHY": "Healthy",
            "VIGOROUS": "Vigorous",
        },
        "trajectory": {
            "STRONG_DECLINE": "Strongly declining",
            "DECLINE": "Declining",
            "STABLE": "Stable",
            "GROWTH": "Growing",
            "STRONG_GROWTH": "Strong growth",
        },
    },
    "fr": {
        "health": {
            "CRITICAL": "Critique",
            "STRESSED": "En détresse",
            "MODERATE": "Modéré",
            "HEALTHY": "Sain",
            "VIGOROUS": "Vigoureux",
        },
        "trajectory": {
            "STRONG_DECLINE": "Déclin marqué",
            "DECLINE": "En déclin",
            "STABLE": "Stable",
            "GROWTH": "En croissance",
            "STRONG_GROWTH": "Forte croissance",
        },
    },
    "ar": {
        "health": {
            "CRITICAL": "حرجة",
            "STRESSED": "تحت ضغط",
            "MODERATE": "متوسطة",
            "HEALTHY": "سليمة",
            "VIGOROUS": "نشطة",
        },
        "trajectory": {
            "STRONG_DECLINE": "تراجع حاد",
            "DECLINE": "في تراجع",
            "STABLE": "مستقرة",
            "GROWTH": "في نمو",
            "STRONG_GROWTH": "نمو قوي",
        },
    },
}


def _t(lang: str, key: str) -> str:
    table = _STRINGS.get(lang, _STRINGS["en"])
    return table.get(key, _STRINGS["en"].get(key, key))


# Legacy free-text trajectory strings written before the enum migration. Used
# to coerce stored values onto the new canonical codes so older reports still
# render with localized chips.
_LEGACY_TRAJECTORY_ALIAS: dict[str, str] = {
    "DECLINING (significant drop across season)": "STRONG_DECLINE",
    "SLIGHTLY DECLINING": "DECLINE",
    "STABLE": "STABLE",
    "GROWING": "GROWTH",
    "STRONG GROWTH": "STRONG_GROWTH",
}


def _code_label(lang: str, kind: str, value: str) -> str:
    table = _CODE_LABELS.get(lang, _CODE_LABELS["en"]).get(kind, {})
    if value in table:
        return table[value]
    if kind == "trajectory":
        canonical = _LEGACY_TRAJECTORY_ALIAS.get(value.strip().upper())
        if canonical and canonical in table:
            return table[canonical]
    return value


def _render_text(text: str, lang: str) -> str:
    if lang == "ar":
        return _shape_arabic(text)
    return text


def _styles(lang: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    is_rtl = lang == "ar"
    font = ARABIC_FONT if is_rtl else LATIN_FONT
    align = TA_RIGHT if is_rtl else TA_LEFT

    return {
        "title": ParagraphStyle(
            "Title",
            parent=base,
            fontName=font,
            fontSize=20,
            leading=26,
            alignment=align,
            spaceAfter=12,
            textColor=colors.HexColor("#0e7c33"),
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base,
            fontName=font,
            fontSize=13,
            leading=18,
            alignment=align,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#1f2937"),
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base,
            fontName=font,
            fontSize=11,
            leading=15,
            alignment=align,
        ),
        "muted": ParagraphStyle(
            "Muted",
            parent=base,
            fontName=font,
            fontSize=9,
            leading=12,
            alignment=align,
            textColor=colors.HexColor("#6b7280"),
        ),
    }


def build_report(analysis: Analysis, lang: str = "en") -> bytes:
    """Render the analysis as a PDF and return the bytes."""
    _ensure_fonts()
    if lang not in _STRINGS:
        lang = "en"
    is_rtl = lang == "ar"
    styles = _styles(lang)
    result = analysis.result
    if result is None:
        raise ValueError("analysis has no result yet")

    def p(key_or_text: str, style: str = "body", *, raw: bool = False) -> Paragraph:
        text = key_or_text if raw else _t(lang, key_or_text)
        return Paragraph(_render_text(text, lang), styles[style])

    def kv(key: str, value: str) -> list:
        label = Paragraph(_render_text(_t(lang, key), lang), styles["body"])
        val = Paragraph(_render_text(value, lang), styles["body"])
        return [label, val] if not is_rtl else [val, label]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=_t(lang, "title"),
    )

    flow: list = []
    flow.append(p("title", "title"))

    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    flow.append(p(f"{_t(lang, 'generated')}: {generated}", "muted", raw=True))
    flow.append(p(f"{_t(lang, 'source')}: {analysis.source_name}", "muted", raw=True))
    flow.append(p(f"{_t(lang, 'analysis_id')}: #{analysis.id}", "muted", raw=True))
    flow.append(Spacer(1, 0.4 * cm))

    map_path = Path(result.map_png_path)
    if map_path.is_file():
        flow.append(p("classification_map", "h2"))
        img = Image(str(map_path))
        max_w = 17 * cm
        max_h = 12 * cm
        ratio = min(max_w / img.imageWidth, max_h / img.imageHeight)
        img.drawWidth = img.imageWidth * ratio
        img.drawHeight = img.imageHeight * ratio
        img.hAlign = "RIGHT" if is_rtl else "LEFT"
        flow.append(img)
        flow.append(Spacer(1, 0.4 * cm))

    flow.append(p("predicted_crop", "h2"))
    summary_rows = [
        kv("predicted_crop", result.predicted_crop.replace("_", " ")),
        kv("health", _code_label(lang, "health", result.health_status)),
        kv("trajectory", _code_label(lang, "trajectory", result.trajectory)),
        kv("confidence", f"{result.mean_confidence * 100:.1f}%"),
        kv("ndvi", f"{result.mean_ndvi:.3f}"),
        kv("ndre", f"{result.mean_ndre:.3f}"),
        kv("savi", f"{result.mean_savi:.3f}"),
        kv("temporal_trend", f"{result.temporal_trend:+.3f}"),
    ]
    summary = Table(summary_rows, colWidths=[6 * cm, 11 * cm])
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f9fafb")),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#e5e7eb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    flow.append(summary)
    flow.append(Spacer(1, 0.4 * cm))

    flow.append(p("advice", "h2"))
    flow.append(p(result.advice, "body", raw=True))
    flow.append(Spacer(1, 0.4 * cm))

    flow.append(PageBreak())

    flow.append(p("class_distribution", "h2"))
    sorted_dist = sorted(result.class_distribution.items(), key=lambda kv: -kv[1])
    header = [_t(lang, "class_label"), _t(lang, "share")]
    if is_rtl:
        header = list(reversed(header))
    rows = [header]
    for cls, frac in sorted_dist:
        cell_label = Paragraph(_render_text(cls.replace("_", " "), lang), styles["body"])
        cell_value = Paragraph(_render_text(f"{frac * 100:.1f}%", lang), styles["body"])
        rows.append([cell_value, cell_label] if is_rtl else [cell_label, cell_value])
    dist_table = Table(rows, colWidths=[12 * cm, 5 * cm])
    dist_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e7c33")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), ARABIC_FONT if is_rtl else LATIN_FONT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    flow.append(dist_table)
    flow.append(Spacer(1, 0.6 * cm))

    flow.append(p("farmer_observation", "h2"))
    if result.farmer_notes:
        flow.append(p(result.farmer_notes, "body", raw=True))
        if result.observed_at:
            stamp = result.observed_at.strftime("%Y-%m-%d")
            flow.append(p(f"{_t(lang, 'observed_on')}: {stamp}", "muted", raw=True))
    else:
        flow.append(p("no_observation", "muted"))

    doc.build(flow)
    return buffer.getvalue()
