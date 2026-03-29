
from __future__ import annotations

import math

def signal_mean_reversion(msv, threshold: float = 2.0) -> float:
    z20 = getattr(msv, "zscore_20", float("nan"))
    z100 = getattr(msv, "zscore_100", float("nan"))
    if math.isnan(z20) or math.isnan(z100):
        return 0.0

    if z20 < -threshold and z100 < -threshold * 0.5:
        return min(1.0, abs(z20) / (threshold * 2))
    if z20 > threshold and z100 > threshold * 0.5:
        return max(-1.0, -abs(z20) / (threshold * 2))
    return 0.0

def signal_momentum(msv, threshold: float = 0.5) -> float:
    r60 = getattr(msv, "ret_60s", float("nan"))
    r300 = getattr(msv, "ret_300s", float("nan"))
    if math.isnan(r60) or math.isnan(r300):
        return 0.0

    composite = 0.6 * r60 + 0.4 * r300
    return max(-1.0, min(1.0, composite / threshold))

def signal_ofi(msv) -> float:
    ofi = getattr(msv, "ofi", float("nan"))
    if math.isnan(ofi):
        return 0.0
    return max(-1.0, min(1.0, ofi * 3.0))

def signal_volume_anomaly(msv, vol_threshold: float = 2.0) -> float:
    vr = getattr(msv, "volume_ratio_20", float("nan"))
    r1s = getattr(msv, "ret_1s", float("nan"))
    if math.isnan(vr) or math.isnan(r1s):
        return 0.0

    if vr < vol_threshold:
        return 0.0

    direction = 1.0 if r1s > 0 else (-1.0 if r1s < 0 else 0.0)
    magnitude = min(1.0, (vr - vol_threshold) / vol_threshold)
    return direction * magnitude

# Registry for iteration
TIER1_SIGNALS = {
    "mean_reversion": signal_mean_reversion,
    "momentum": signal_momentum,
    "ofi": signal_ofi,
    "volume_anomaly": signal_volume_anomaly,
}
