"""S-band link budget for EventSat (closure / margin).

Computes downlink and uplink received power and margin from the component
parameters (Alén Space ground station + EnduroSat S-band hardware; see the PDR
link-budget tables, mirrored in ``configs/scenarios/eventsat.yaml`` under
``communications.link_budget``). The link is evaluated at the worst-case geometry
— the longest slant range, when the satellite is at the minimum elevation over the
ground station's horizon.

This is a *closure* model: it tells you whether the RF link is viable and with what
margin. The effective data rate is NOT set here — it is capped by the OBC→S-band
transmitter protocol (50 kbps), well below the RF capability, so the rate is a
configured constant (``communications.sband.downlink_rate_kbps``) and this module
only validates/documents that the channel closes.

All quantities in dB / dBm. Pure functions (no side effects).
"""
from __future__ import annotations

import math
from typing import Any, Dict

_EARTH_RADIUS_KM = 6371.0


def slant_range_km(altitude_km: float, elevation_deg: float) -> float:
    """Slant range to a satellite at ``altitude_km`` seen at ``elevation_deg``
    over a spherical Earth (longest at the minimum elevation)."""
    re = _EARTH_RADIUS_KM
    eps = math.radians(elevation_deg)
    return math.sqrt((re * math.sin(eps)) ** 2 + 2 * re * altitude_km + altitude_km ** 2) \
        - re * math.sin(eps)


def free_space_loss_db(distance_km: float, frequency_mhz: float) -> float:
    """Free-space path loss (dB)."""
    return 20 * math.log10(distance_km) + 20 * math.log10(frequency_mhz) + 32.44


def compute_link_budget(
    lb: Dict[str, Any], altitude_km: float, elevation_deg: float
) -> Dict[str, float]:
    """Return downlink/uplink received power + margin at the given geometry.

    Args:
        lb: the ``communications.link_budget`` config block (downlink/uplink/losses).
        altitude_km: satellite altitude.
        elevation_deg: elevation over the GS horizon (use the minimum for worst case).

    Returns dict with slant_range_km, fspl_*_db, *_prx_dbm, *_margin_db.
    """
    d_km = slant_range_km(altitude_km, elevation_deg)
    losses = lb.get("losses", {})
    extra = losses.get("atmosphere_db", 2.0) + losses.get("pointing_error_db", 2.0)

    dl = lb.get("downlink", {})
    fspl_dl = free_space_loss_db(d_km, dl.get("frequency_mhz", 2245.0))
    eirp_sat = dl.get("sat_tx_power_dbm", 33.0) + dl.get("sat_antenna_gain_dbi", 5.0) \
        - dl.get("sat_cable_loss_db", 2.0)
    prx_gs = eirp_sat - fspl_dl - extra + dl.get("gs_antenna_gain_dbi", 31.34) \
        - dl.get("gs_cable_loss_db", 2.0)
    dl_margin = prx_gs - dl.get("gs_sensitivity_dbm", -100.0)

    ul = lb.get("uplink", {})
    fspl_ul = free_space_loss_db(d_km, ul.get("frequency_mhz", 2067.5))
    eirp_gs = ul.get("gs_tx_power_dbm", 44.7) + ul.get("gs_pa_gain_db", 33.0) \
        + ul.get("gs_antenna_gain_dbi", 31.34) - ul.get("gs_cable_loss_db", 2.0)
    prx_sat = eirp_gs - fspl_ul - extra + ul.get("sat_antenna_gain_dbi", 5.0) \
        - ul.get("sat_cable_loss_db", 2.0)
    ul_margin = prx_sat - ul.get("sat_sensitivity_dbm", -121.0)

    return {
        "slant_range_km": d_km,
        "fspl_downlink_db": fspl_dl,
        "fspl_uplink_db": fspl_ul,
        "downlink_prx_dbm": prx_gs,
        "downlink_margin_db": dl_margin,
        "uplink_prx_dbm": prx_sat,
        "uplink_margin_db": ul_margin,
    }
