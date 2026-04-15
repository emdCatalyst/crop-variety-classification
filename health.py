_GENERIC_ADVICE = {
    "CRITICAL":  "Immediate field inspection required. Potential crop failure, severe water deficit, or disease outbreak.",
    "STRESSED":  "Check irrigation schedule and soil moisture. Inspect for pest or disease pressure. Consider foliar nutrient application.",
    "MODERATE":  "Monitor closely over the next 7-10 days. Verify fertilisation plan is on schedule.",
    "HEALTHY":   "Crop developing normally. Continue standard management protocol.",
    "VIGOROUS":  "Excellent canopy development. Verify no lodging risk for dense-canopy crops.",
}

_CROP_ADVICE = {
    "Maize":     {"STRESSED":  "Maize NDRE decline often precedes visible wilting by 5-7 days. Check soil moisture at 30 cm depth and inspect for rootworm damage.",
                  "CRITICAL":  "Severe Maize stress. Assess for Northern Corn Leaf Blight or Gray Leaf Spot if conditions are humid."},
    "Wheat":     {"STRESSED":  "Wheat NDRE drop frequently indicates nitrogen deficiency. Consider topdressing (30-40 kg N/ha) if at tillering or stem extension stage.",
                  "MODERATE":  "Verify fungicide schedule if humid conditions persist — Septoria risk is elevated."},
    "Potato":    {"STRESSED":  "Check irrigation uniformity and Late Blight (Phytophthora) pressure. Potato canopy collapse is rapid once disease spreads.",
                  "CRITICAL":  "Severe Potato stress — inspect immediately for Late Blight. Full canopy loss can occur within 7 days under humid conditions."},
    "Soybean":   {"STRESSED":  "NDRE drop may indicate iron deficiency chlorosis (IDC) or soybean cyst nematode. Check nodulation status at root level."},
    "SugarBeet": {"STRESSED":  "Check for Cercospora leaf spot and verify soil pH. Low NDRE in SugarBeet correlates with nitrogen or boron deficiency.",
                  "MODERATE":  "Monitor leaf area index. SugarBeet is sensitive to water stress during the root-filling phase."},
    "Intercrop": {"MODERATE":  "Intercrop canopy competition naturally lowers per-species NDVI. Assess species balance rather than total NDVI alone.",
                  "STRESSED":  "One intercrop component may be suppressing the other. Inspect competitive balance between species."},
}

def assess_health(ndvi_mean, ndre_mean, savi_mean, crop_species, temporal_trend):
    """
    Returns a health dict using established NDVI/NDRE thresholds.

    Parameters
    ----------
    ndvi_mean      : float  — mean NDVI across all timesteps and patch pixels
    ndre_mean      : float  — mean NDRE across all timesteps and patch pixels
    savi_mean      : float  — mean SAVI across all timesteps and patch pixels
    crop_species   : str    — e.g. "Maize", "Wheat" (first part of class name)
    temporal_trend : float  — ndvi_last_date - ndvi_first_date (+ = growing, - = declining)
    """
    # Primary status from NDVI
    if ndvi_mean < 0.20:
        status = "CRITICAL"
    elif ndvi_mean < 0.35:
        status = "STRESSED"
    elif ndvi_mean < 0.55:
        # Use NDRE to refine: it catches early chlorophyll stress that NDVI misses
        status = "STRESSED" if ndre_mean < 0.25 else "MODERATE"
    elif ndvi_mean < 0.75:
        status = "HEALTHY"
    else:
        status = "VIGOROUS"

    # Temporal trajectory
    if temporal_trend < -0.10:
        trajectory = "DECLINING ⚠ (significant drop across season)"
    elif temporal_trend < -0.03:
        trajectory = "SLIGHTLY DECLINING"
    elif temporal_trend > 0.10:
        trajectory = "STRONG GROWTH ✓"
    elif temporal_trend > 0.03:
        trajectory = "GROWING ✓"
    else:
        trajectory = "STABLE"

    # Pick most specific advice available, fall back to generic
    advice = (_CROP_ADVICE.get(crop_species, {}).get(status)
              or _GENERIC_ADVICE[status])

    return {
        "ndvi_mean":     round(float(ndvi_mean),  3),
        "ndre_mean":     round(float(ndre_mean),  3),
        "savi_mean":     round(float(savi_mean),  3),
        "health_status": status,
        "trajectory":    trajectory,
        "advice":        advice,
    }
