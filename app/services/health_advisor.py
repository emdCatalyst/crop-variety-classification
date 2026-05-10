"""Pluggable agronomy-expert assessment for crop health.

Wraps an external LLM (Google Gemini or Groq Llama) or the in-repo heuristic
to produce a HealthAssessment for a freshly inferred crop analysis. The
provider is chosen via settings.health_ai_provider; ANY LLM failure (timeout,
malformed JSON, missing key, network error) automatically falls back to the
heuristic so the inference pipeline never blocks on an outage.

The LLM is asked to write the free-form ``advice`` field in the user's UI
language, while ``health_status`` and ``trajectory`` always come back as
canonical English enum codes — the frontend and PDF renderer translate those
codes at display time so they follow whatever language the viewer has active.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from ..core.config import Settings, get_settings
from ..schemas.health import HealthAssessment, HealthStatus, Trajectory

log = logging.getLogger("agrovision.health_advisor")


@dataclass
class HealthContext:
    analysis_id: int
    predicted_crop: str
    class_distribution: dict[str, float]
    mean_confidence: float
    mean_ndvi: float
    mean_ndre: float
    mean_savi: float
    temporal_trend: float
    heuristic_health_status: str
    heuristic_trajectory: str
    heuristic_advice: str
    region: str | None = None
    season: str | None = None
    # User's UI language at the moment the analysis is being generated.
    # The LLM writes ``advice`` in this language; categorical fields stay as
    # English enum codes (translated client-side).
    language: str = "en"


_VALID_STATUSES: tuple[HealthStatus, ...] = (
    "CRITICAL",
    "STRESSED",
    "MODERATE",
    "HEALTHY",
    "VIGOROUS",
)
_VALID_TRAJECTORIES: tuple[Trajectory, ...] = (
    "STRONG_DECLINE",
    "DECLINE",
    "STABLE",
    "GROWTH",
    "STRONG_GROWTH",
)
_LANG_NAMES: dict[str, str] = {"en": "English", "fr": "French", "ar": "Arabic"}


def _coerce_status(raw: str) -> HealthStatus:
    cleaned = (raw or "").strip().upper()
    return cleaned if cleaned in _VALID_STATUSES else "MODERATE"  # type: ignore[return-value]


def _coerce_trajectory(raw: str) -> Trajectory:
    cleaned = (raw or "").strip().upper().replace(" ", "_").replace("-", "_")
    return cleaned if cleaned in _VALID_TRAJECTORIES else "STABLE"  # type: ignore[return-value]


# Maps the inference_core's free-text trajectory string ("SLIGHTLY DECLINING",
# "STRONG GROWTH", etc.) to the new canonical enum code. Used by the heuristic
# fallback, which still receives the long-form string from inference_core.
_HEURISTIC_TRAJECTORY_MAP: dict[str, Trajectory] = {
    "DECLINING (significant drop across season)": "STRONG_DECLINE",
    "SLIGHTLY DECLINING": "DECLINE",
    "STABLE": "STABLE",
    "GROWING": "GROWTH",
    "STRONG GROWTH": "STRONG_GROWTH",
}


# Localized canned advice for the heuristic fallback path. Used only when the
# LLM is unavailable or the provider is set to "heuristic".
_HEURISTIC_ADVICE: dict[str, dict[HealthStatus, str]] = {
    "en": {
        "CRITICAL": "Crop is in critical condition. Inspect the field immediately for water stress, nutrient deficiency, or disease pressure, and consider emergency irrigation or remediation.",
        "STRESSED": "Crop shows clear stress signals. Check soil moisture and recent nitrogen status; scout for pests and adjust irrigation or fertilization within the next few days.",
        "MODERATE": "Crop is in average condition. Maintain the current schedule but tighten scouting cadence; small interventions now can prevent decline.",
        "HEALTHY": "Crop is in good shape. Keep current irrigation and nutrient schedule; routine scouting is sufficient.",
        "VIGOROUS": "Crop is vigorous. Maintain current practices and monitor for any over-canopy issues such as lodging risk or excessive humidity.",
    },
    "fr": {
        "CRITICAL": "La culture est en état critique. Inspectez le champ immédiatement pour stress hydrique, carence nutritionnelle ou pression sanitaire, et envisagez une irrigation ou une remédiation d'urgence.",
        "STRESSED": "La culture montre des signes clairs de stress. Vérifiez l'humidité du sol et le statut azoté récent ; surveillez les ravageurs et ajustez l'irrigation ou la fertilisation dans les jours qui viennent.",
        "MODERATE": "La culture est en condition moyenne. Maintenez le programme en cours mais resserrez le rythme de surveillance ; de petites interventions maintenant peuvent éviter un déclin.",
        "HEALTHY": "La culture est en bonne forme. Conservez l'irrigation et la fertilisation actuelles ; une surveillance de routine suffit.",
        "VIGOROUS": "La culture est vigoureuse. Maintenez les pratiques actuelles et surveillez tout problème lié à un couvert trop dense (risque de verse, humidité excessive).",
    },
    "ar": {
        "CRITICAL": "المحصول في حالة حرجة. افحص الحقل فوراً بحثاً عن إجهاد مائي أو نقص في العناصر أو ضغط مرضي، وفكّر في ريّ أو تدخل طارئ.",
        "STRESSED": "تُظهر المحصول علامات إجهاد واضحة. تحقّق من رطوبة التربة وحالة النيتروجين الأخيرة؛ افحص الآفات واضبط الريّ أو التسميد خلال الأيام القادمة.",
        "MODERATE": "المحصول في حالة متوسطة. حافظ على البرنامج الحالي مع تكثيف وتيرة المتابعة؛ قد تمنع تدخّلات صغيرة الآن أيّ تدهور.",
        "HEALTHY": "المحصول في حالة جيدة. احتفظ ببرنامج الريّ والتسميد الحاليّين؛ تكفي المتابعة الاعتيادية.",
        "VIGOROUS": "المحصول قوي النمو. حافظ على الممارسات الحالية وراقب أي مشاكل ناتجة عن كثافة المجموع الخضري (خطر الانكفاء، الرطوبة الزائدة).",
    },
}


def _heuristic_assessment(ctx: HealthContext) -> HealthAssessment:
    status = _coerce_status(ctx.heuristic_health_status)
    trajectory = _HEURISTIC_TRAJECTORY_MAP.get(
        (ctx.heuristic_trajectory or "").strip().upper(), "STABLE"
    )
    lang = ctx.language if ctx.language in _HEURISTIC_ADVICE else "en"
    advice = _HEURISTIC_ADVICE[lang][status]
    return HealthAssessment(
        health_status=status,
        trajectory=trajectory,
        advice=advice,
        confidence=1.0,
        citations=[],
        source="heuristic",
    )


def _build_prompt(ctx: HealthContext) -> str:
    top = sorted(ctx.class_distribution.items(), key=lambda kv: -kv[1])[:5]
    top_str = "; ".join(f"{cls.replace('_', ' ')} {frac * 100:.1f}%" for cls, frac in top)
    region_line = ""
    if ctx.region or ctx.season:
        region_line = f"\n- Region/season: {ctx.region or 'unknown'} / {ctx.season or 'unknown'}"
    lang_name = _LANG_NAMES.get(ctx.language, "English")
    return (
        "You are an expert agronomist assessing a crop plot from multi-temporal "
        "multispectral UAV imagery. Use the measurements below to determine plot "
        "health and recommend specific, actionable agronomic interventions for the "
        "farmer (irrigation timing, nitrogen application, scouting for pests/disease, "
        "drainage, harvest timing, etc.). Avoid generic advice.\n\n"
        f"Write the `advice` field in {lang_name}. The `health_status` and "
        "`trajectory` fields must use the canonical English codes listed below — "
        "do NOT translate them.\n\n"
        "Plot measurements:\n"
        f"- Predicted crop / variety: {ctx.predicted_crop.replace('_', ' ')}\n"
        f"- Top class shares: {top_str}\n"
        f"- Model classification confidence: {ctx.mean_confidence * 100:.1f}%\n"
        f"- Mean NDVI: {ctx.mean_ndvi:.3f} (typical healthy range 0.5-0.9)\n"
        f"- Mean NDRE: {ctx.mean_ndre:.3f} (red-edge NDVI; healthy 0.3-0.5; sensitive to N stress)\n"
        f"- Mean SAVI: {ctx.mean_savi:.3f} (soil-adjusted NDVI)\n"
        f"- Temporal NDVI trend across 12 timesteps: {ctx.temporal_trend:+.3f}"
        f"{region_line}\n\n"
        "Respond with strict JSON ONLY (no prose, no markdown fences) matching:\n"
        '{"health_status": "<CRITICAL|STRESSED|MODERATE|HEALTHY|VIGOROUS>",\n'
        ' "trajectory":    "<STRONG_DECLINE|DECLINE|STABLE|GROWTH|STRONG_GROWTH>",\n'
        f' "advice":        "2-4 concise sentences with specific actions, written in {lang_name}",\n'
        ' "confidence":    0.0-1.0,\n'
        ' "citations":     ["short reference", ...]}'
    )


_GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "health_status": {
            "type": "string",
            "enum": list(_VALID_STATUSES),
        },
        "trajectory": {
            "type": "string",
            "enum": list(_VALID_TRAJECTORIES),
        },
        "advice": {"type": "string"},
        "confidence": {"type": "number"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["health_status", "trajectory", "advice"],
    "propertyOrdering": ["health_status", "trajectory", "advice", "confidence", "citations"],
}


async def _call_gemini(prompt: str, settings: Settings) -> str:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _GEMINI_RESPONSE_SCHEMA,
            "temperature": 0.0,
        },
    }
    async with httpx.AsyncClient(timeout=settings.health_ai_timeout_s) as client:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"x-goog-api-key": settings.gemini_api_key},
                )
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"Gemini HTTP {resp.status_code}: {resp.text[:600]}"
                    )
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (httpx.HTTPError, RuntimeError, KeyError, IndexError) as exc:
                last_exc = exc
                if attempt == 0:
                    log.warning("Gemini attempt 1 failed: %s; retrying", exc)
                    continue
        raise last_exc or RuntimeError("Gemini call exhausted retries")


async def _call_groq(prompt: str, settings: Settings) -> str:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured")
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": settings.groq_model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=settings.health_ai_timeout_s) as client:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                )
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"Groq HTTP {resp.status_code}: {resp.text[:600]}"
                    )
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, RuntimeError, KeyError, IndexError) as exc:
                last_exc = exc
                if attempt == 0:
                    log.warning("Groq attempt 1 failed: %s; retrying", exc)
                    continue
        raise last_exc or RuntimeError("Groq call exhausted retries")


def _parse_response(raw_json: str, source: str) -> HealthAssessment:
    parsed = json.loads(raw_json)
    return HealthAssessment(
        health_status=_coerce_status(str(parsed.get("health_status", ""))),
        trajectory=_coerce_trajectory(str(parsed.get("trajectory", ""))),
        advice=str(parsed.get("advice", "")).strip()[:2000] or "No advice available.",
        confidence=float(parsed.get("confidence", 0.7)),
        citations=[str(c) for c in (parsed.get("citations") or [])][:10],
        source=source,
    )


async def assess(ctx: HealthContext) -> HealthAssessment:
    """Produce a HealthAssessment for the given context.

    Never raises: any failure path falls back to the heuristic assessment.
    """
    settings = get_settings()
    provider = (settings.health_ai_provider or "heuristic").lower()

    if provider == "heuristic":
        return _heuristic_assessment(ctx)

    if provider not in {"gemini", "groq"}:
        log.warning("Unknown HEALTH_AI_PROVIDER=%s; falling back to heuristic", provider)
        return _heuristic_assessment(ctx)

    prompt = _build_prompt(ctx)
    try:
        raw = await (_call_gemini(prompt, settings) if provider == "gemini" else _call_groq(prompt, settings))
        return _parse_response(raw, provider)
    except (httpx.HTTPError, json.JSONDecodeError, ValidationError, KeyError, ValueError, RuntimeError) as exc:
        log.warning(
            "health_ai provider=%s failed for analysis %s: %s: %s; falling back to heuristic",
            provider,
            ctx.analysis_id,
            type(exc).__name__,
            exc,
        )
        return _heuristic_assessment(ctx)
