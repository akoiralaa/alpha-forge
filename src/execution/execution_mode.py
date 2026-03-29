"""Passive vs aggressive execution decision.

Decides whether to use passive (limit) or aggressive (market) execution
based on signal half-life, fill time estimates, VPIN, and regime.
"""

from __future__ import annotations


def execution_mode(
    signal_score: float,
    signal_half_life_s: float,
    estimated_passive_fill_time_s: float,
    current_vpin: float,
    vpin_passive_threshold: float = 0.90,
    regime_execution_mode: str = "PASSIVE_PREFERRED",
) -> str:
    """Decide execution mode: 'PASSIVE', 'AGGRESSIVE', or 'BLOCK'.

    Args:
        signal_score: Current signal strength [-1, 1].
        signal_half_life_s: Estimated seconds until signal decays to half.
        estimated_passive_fill_time_s: Estimated time to fill a passive order.
        current_vpin: Current VPIN value [0, 1].
        vpin_passive_threshold: VPIN level above which only passive allowed.
        regime_execution_mode: From REGIME_PARAMS — PASSIVE_PREFERRED, PASSIVE_ONLY, REDUCE_ONLY.

    Returns:
        'PASSIVE', 'AGGRESSIVE', or 'BLOCK'
    """
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
