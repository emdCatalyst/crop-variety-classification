"""Pluggable agronomy-expert assessment for crop health.

Wraps an external LLM (Google Gemini or Groq Llama) or the in-repo heuristic
to produce a HealthAssessment for a freshly inferred crop analysis. The
provider is chosen via settings.health_ai_provider; ANY LLM failure (timeout,
malformed JSON, missing key, network error) automatically falls back to the
heuristic so the inference pipeline never blocks on an outage.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from ..core.config import Settings, get_settings
from ..schemas.health import HealthAssessment, HealthStatus

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


_VALID_STATUSES: tuple[HealthStatus, ...] = (
    "CRITICAL",
    "STRESSED",
    "MODERATE",
    "HEALTHY",
    "VIGOROUS",
)


def _coerce_status(raw: str) -> HealthStatus:
    cleaned = (raw or "").strip().upper()
    return cleaned if cleaned in _VALID_STATUSES else "MODERATE"  # type: ignore[return-value]


def _heuristic_assessment(ctx: HealthContext) -> HealthAssessment:
    return HealthAssessment(
        health_status=_coerce_status(ctx.heuristic_health_status),
        trajectory=ctx.heuristic_trajectory or "Unknown",
        advice=ctx.heuristic_advice or "No advice available.",
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
    return (
        "You are an expert agronomist assessing a crop plot from multi-temporal "
        "multispectral UAV imagery. Use the measurements below to determine plot health "
        "and recommend specific, actionable agronomic interventions for the farmer "
        "(irrigation timing, nitrogen application, scouting for pests/disease, drainage, "
        "harvest timing, etc.). Avoid generic advice.\n\n"
        "Plot measurements:\n"
        f"- Predicted crop / variety: {ctx.predicted_crop.replace('_', ' ')}\n"
        f"- Top class shares: {top_str}\n"
        f"- Model classification confidence: {ctx.mean_confidence * 100:.1f}%\n"
        f"- Mean NDVI: {ctx.mean_ndvi:.3f} (typical healthy range 0.5-0.9)\n"
        f"- Mean NDRE: {ctx.mean_ndre:.3f} (red-edge NDVI; healthy 0.3-0.5; sensitive to N stress)\n"
        f"- Mean SAVI: {ctx.mean_savi:.3f} (soil-adjusted NDVI)\n"
        f"- Temporal NDVI trend across 12 timesteps: {ctx.temporal_trend:+.3f}"
        f"{region_line}\n\n"
        "Heuristic baseline (do NOT just echo it; correct it where measurements warrant):\n"
        f"- baseline_status: {ctx.heuristic_health_status}\n"
        f"- baseline_trajectory: {ctx.heuristic_trajectory}\n\n"
        "Respond with strict JSON ONLY (no prose, no markdown fences) matching:\n"
        '{"health_status": "<CRITICAL|STRESSED|MODERATE|HEALTHY|VIGOROUS>",\n'
        ' "trajectory": "short phrase, max ~120 chars",\n'
        ' "advice": "2-4 concise sentences with specific actions",\n'
        ' "confidence": 0.0-1.0,\n'
        ' "citations": ["short reference", ...]}'
    )


_GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "health_status": {
            "type": "string",
            "enum": list(_VALID_STATUSES),
        },
        "trajectory": {"type": "string"},
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
            "temperature": 0.3,
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
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (httpx.HTTPError, KeyError, IndexError) as exc:
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
        "temperature": 0.3,
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
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_exc = exc
                if attempt == 0:
                    log.warning("Groq attempt 1 failed: %s; retrying", exc)
                    continue
        raise last_exc or RuntimeError("Groq call exhausted retries")


def _parse_response(raw_json: str, source: str) -> HealthAssessment:
    parsed = json.loads(raw_json)
    return HealthAssessment(
        health_status=_coerce_status(str(parsed.get("health_status", ""))),
        trajectory=str(parsed.get("trajectory", "")).strip()[:200] or "Unknown",
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
            "health_ai provider=%s failed for analysis %s: %s; falling back to heuristic",
            provider,
            ctx.analysis_id,
            exc,
        )
        return _heuristic_assessment(ctx)
