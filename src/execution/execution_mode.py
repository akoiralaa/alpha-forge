
from __future__ import annotations

def execution_mode(
    signal_score: float,
    signal_half_life_s: float,
    estimated_passive_fill_time_s: float,
    current_vpin: float,
    vpin_passive_threshold: float = 0.90,
    regime_execution_mode: str = "PASSIVE_PREFERRED",
) -> str:
    if regime_execution_mode == "REDUCE_ONLY":
        # Only reduce-direction fills allowed via passive
        return "BLOCK" if signal_score > 0 else "PASSIVE"

    if regime_execution_mode == "PASSIVE_ONLY":
        return "PASSIVE"

    if current_vpin > vpin_passive_threshold:
        return "PASSIVE"

    # Half-life shorter than fill time: edge will decay before passive fill → go aggressive
    if signal_half_life_s < estimated_passive_fill_time_s * 1.5:
        return "AGGRESSIVE"

    return "PASSIVE"
