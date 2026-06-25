"""Inter-satellite-link primitives for SSA.

Ports the UHF/QPSK communication math from the autops-rl
``CommSubsystem``: free-space loss -> SNR -> ideal rate -> BER ->
effective rate. Distances are explicit and pure; no simulator state lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


_BOLTZMANN_J_PER_K = 1.38e-23
_LIGHT_SPEED_M_S = 3.0e8


@dataclass(frozen=True)
class ISLConfig:
    """UHF/QPSK link-budget constants from autops-rl ``CommSubsystem``."""

    tx_power_w: float = 2.0
    rx_gain_db: float = 1.0
    rx_loss_db: float = 0.5
    tx_gain_db: float = 1.0
    tx_loss_db: float = 3.0
    frequency_hz: float = 437e6
    bandwidth_hz: float = 9600.0
    symbol_rate_hz: float = 9600.0
    modulation_order: int = 4
    sensitivity_dbw: float = -151.0
    noise_temperature_k: float = 290.0


def vector_range_km(a_km: Sequence[float], b_km: Sequence[float]) -> float:
    """Return Euclidean range between two 3D vectors in km."""

    if len(a_km) != 3 or len(b_km) != 3:
        raise ValueError("vector_range_km expects two 3D vectors")
    return math.sqrt(sum((float(b) - float(a)) ** 2 for a, b in zip(a_km, b_km)))


def free_space_loss_db(distance_m: float, frequency_hz: float = 437e6) -> float:
    """Free-space path loss for distance in meters and frequency in Hz."""

    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive")
    return 20.0 * math.log10(4.0 * math.pi * distance_m * frequency_hz / _LIGHT_SPEED_M_S)


def noise_power_dbw(config: ISLConfig | None = None) -> float:
    """Thermal noise power, kTB, in dBW."""

    cfg = config or ISLConfig()
    return 10.0 * math.log10(_BOLTZMANN_J_PER_K * cfg.noise_temperature_k * cfg.bandwidth_hz)


def received_power_dbw(distance_m: float, config: ISLConfig | None = None) -> float:
    """Received power in dBW using the autops-rl ISL budget."""

    cfg = config or ISLConfig()
    tx_power_dbw = 10.0 * math.log10(cfg.tx_power_w)
    return (
        tx_power_dbw
        + cfg.rx_gain_db
        + cfg.tx_gain_db
        - cfg.rx_loss_db
        - cfg.tx_loss_db
        - free_space_loss_db(distance_m, cfg.frequency_hz)
    )


def snr_db(distance_m: float, config: ISLConfig | None = None) -> float:
    """Signal-to-noise ratio in dB."""

    cfg = config or ISLConfig()
    return received_power_dbw(distance_m, cfg) - noise_power_dbw(cfg)


def ideal_data_rate_bps(distance_m: float, config: ISLConfig | None = None) -> float:
    """Shannon rate capped by the configured QPSK symbol rate."""

    cfg = config or ISLConfig()
    snr_linear = 10.0 ** (snr_db(distance_m, cfg) / 10.0)
    shannon_rate = cfg.bandwidth_hz * math.log2(1.0 + snr_linear)
    modulation_cap = cfg.symbol_rate_hz * math.log2(cfg.modulation_order)
    return min(shannon_rate, modulation_cap)


def bit_error_rate(distance_m: float, config: ISLConfig | None = None) -> float:
    """QPSK BER approximation from autops-rl ``CommSubsystem.calculateBER``."""

    cfg = config or ISLConfig()
    bitrate = cfg.symbol_rate_hz * math.log2(cfg.modulation_order)
    efficiency = bitrate / cfg.bandwidth_hz if cfg.bandwidth_hz > 0 else 1.0
    ebn0 = 10.0 ** (snr_db(distance_m, cfg) / 10.0) / efficiency
    return (1.0 / math.log2(cfg.modulation_order)) * math.erfc(math.sqrt(2.0 * ebn0))


def effective_data_rate_bps(distance_m: float, config: ISLConfig | None = None) -> float:
    """Effective bit rate after sensitivity and BER losses."""

    cfg = config or ISLConfig()
    if received_power_dbw(distance_m, cfg) < cfg.sensitivity_dbw:
        return 0.0
    return ideal_data_rate_bps(distance_m, cfg) * (1.0 - bit_error_rate(distance_m, cfg))


def link_budget(distance_m: float, config: ISLConfig | None = None) -> dict[str, float]:
    """Return the full scalar ISL budget for diagnostics and tests."""

    cfg = config or ISLConfig()
    prx = received_power_dbw(distance_m, cfg)
    return {
        "distance_m": float(distance_m),
        "free_space_loss_db": free_space_loss_db(distance_m, cfg.frequency_hz),
        "received_power_dbw": prx,
        "sensitivity_dbw": cfg.sensitivity_dbw,
        "margin_db": prx - cfg.sensitivity_dbw,
        "noise_power_dbw": noise_power_dbw(cfg),
        "snr_db": snr_db(distance_m, cfg),
        "ideal_data_rate_bps": ideal_data_rate_bps(distance_m, cfg),
        "ber": bit_error_rate(distance_m, cfg),
        "effective_data_rate_bps": effective_data_rate_bps(distance_m, cfg),
    }


def is_isl_feasible(
    endpoint_a_km: Sequence[float],
    endpoint_b_km: Sequence[float],
    *,
    endpoint_a_idle: bool = True,
    endpoint_b_idle: bool = True,
    config: ISLConfig | None = None,
) -> bool:
    """Return whether an ISL can close and both radios are available."""

    if not endpoint_a_idle or not endpoint_b_idle:
        return False
    distance_m = vector_range_km(endpoint_a_km, endpoint_b_km) * 1000.0
    cfg = config or ISLConfig()
    return received_power_dbw(distance_m, cfg) >= cfg.sensitivity_dbw
