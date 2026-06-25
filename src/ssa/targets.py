"""RSO target catalog, propagation, and optical access helpers for SSA.

The catalog is synthetic randomized SSO, matching the approved SSA hand-off:
fixed M, paired seeds, 600-900 km altitude, near-circular orbits, and uniform
RAAN/argument/true-anomaly draws. Orekit propagation is used when available via
``src.orbital.propagator``; a deterministic two-body fallback keeps local tests
cheap and offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
import random
from typing import Iterable, Mapping, Sequence

from src.orbital import propagator


_EARTH_RADIUS_KM = 6371.0
_MU_EARTH_KM3_S2 = propagator.MU_EARTH / 1.0e9
_DEFAULT_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class RSOTarget:
    """Synthetic resident-space-object orbit definition."""

    object_id: str
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    true_anomaly_deg: float
    size_m: float = 1.0
    priority: float = 1.0
    epoch: datetime = field(default_factory=lambda: _DEFAULT_EPOCH)


@dataclass(frozen=True)
class DetectionAccess:
    """A target inside the anti-nadir FOV and optical range."""

    object_id: str
    position_km: tuple[float, float, float]
    range_km: float
    angle_deg: float
    quality: float


def diffraction_limited_range_km(
    *,
    object_size_m: float = 1.0,
    aperture_diameter_m: float = 0.09,
    wavelength_m: float = 700e-9,
) -> float:
    """Maximum diffraction-limited detection range in km.

    Mirrors autops-rl ``OpticPayload.dist_detect``: D_max = a*d/(2.44*lambda).
    Defaults produce 52.7 km for a 1 m object, 9 cm aperture, and 700 nm light.
    """

    return object_size_m * aperture_diameter_m / (2.44 * wavelength_m) / 1000.0


def generate_sso_catalog(
    count: int,
    *,
    seed: int | None = None,
    altitude_range_km: tuple[float, float] = (600.0, 900.0),
    eccentricity_max: float = 0.001,
    inclination_range_deg: tuple[float, float] = (97.0, 99.0),
    object_size_m: float = 1.0,
    epoch: datetime = _DEFAULT_EPOCH,
) -> list[RSOTarget]:
    """Build a fixed-size synthetic randomized-SSO RSO catalog."""

    rng = random.Random(seed)
    targets: list[RSOTarget] = []
    for idx in range(int(count)):
        altitude_km = rng.uniform(*altitude_range_km)
        targets.append(
            RSOTarget(
                object_id=f"rso_{idx}",
                semi_major_axis_km=_EARTH_RADIUS_KM + altitude_km,
                eccentricity=rng.uniform(0.0, eccentricity_max),
                inclination_deg=rng.uniform(*inclination_range_deg),
                raan_deg=rng.uniform(0.0, 360.0),
                arg_perigee_deg=rng.uniform(0.0, 360.0),
                true_anomaly_deg=rng.uniform(0.0, 360.0),
                size_m=object_size_m,
                priority=1.0,
                epoch=epoch,
            )
        )
    return targets


def propagate_rso_position_km(
    target: RSOTarget,
    epoch_seconds: float,
    *,
    prefer_orekit: bool = True,
) -> tuple[float, float, float]:
    """Propagate a target and return ECI position in km."""

    if prefer_orekit and propagator.is_available():
        try:
            prop = propagator.create_j2_propagator(
                a_km=target.semi_major_axis_km,
                e=target.eccentricity,
                i_deg=target.inclination_deg,
                raan_deg=target.raan_deg,
                argp_deg=target.arg_perigee_deg,
                ta_deg=target.true_anomaly_deg,
                epoch=target.epoch,
            )
            when = target.epoch + timedelta(seconds=float(epoch_seconds))
            return propagator.propagate_position_km(prop, when)
        except Exception:
            # Keep unit tests and cheap smoke runs deterministic even when Orekit
            # is installed but rejects a synthetic near-circular draw.
            pass
    return _propagate_two_body_position_km(target, epoch_seconds)


def propagated_catalog_positions_km(
    targets: Iterable[RSOTarget],
    epoch_seconds: float,
    *,
    prefer_orekit: bool = True,
) -> dict[str, tuple[float, float, float]]:
    """Propagate all targets into an object_id -> position map."""

    return {
        target.object_id: propagate_rso_position_km(
            target, epoch_seconds, prefer_orekit=prefer_orekit
        )
        for target in targets
    }


def anti_nadir_unit(position_km: Sequence[float]) -> tuple[float, float, float]:
    """Return the outward anti-nadir boresight for an ECI position vector."""

    return _unit(position_km)


def detect_targets_in_fov(
    observer_position_km: Sequence[float],
    target_positions_km: Mapping[str, Sequence[float]],
    *,
    fov_half_angle_deg: float = 5.0,
    max_range_km: float | None = None,
) -> list[DetectionAccess]:
    """Return every target inside the anti-nadir cone and optical range."""

    max_range = diffraction_limited_range_km() if max_range_km is None else float(max_range_km)
    boresight = anti_nadir_unit(observer_position_km)
    detections: list[DetectionAccess] = []
    for object_id, target_position in target_positions_km.items():
        rel = tuple(float(t) - float(o) for o, t in zip(observer_position_km, target_position))
        range_km = _norm(rel)
        if range_km <= 0.0 or range_km > max_range:
            continue
        rel_unit = _unit(rel)
        cos_angle = max(-1.0, min(1.0, _dot(boresight, rel_unit)))
        angle_deg = math.degrees(math.acos(cos_angle))
        if angle_deg > fov_half_angle_deg:
            continue
        angle_score = max(0.0, 1.0 - angle_deg / fov_half_angle_deg)
        range_score = max(0.0, 1.0 - range_km / max_range)
        quality = 0.5 * angle_score + 0.5 * range_score
        detections.append(
            DetectionAccess(
                object_id=str(object_id),
                position_km=tuple(float(x) for x in target_position),
                range_km=range_km,
                angle_deg=angle_deg,
                quality=quality,
            )
        )
    return sorted(detections, key=lambda item: item.object_id)


def _propagate_two_body_position_km(target: RSOTarget, epoch_seconds: float) -> tuple[float, float, float]:
    a = target.semi_major_axis_km
    e = target.eccentricity
    n = math.sqrt(_MU_EARTH_KM3_S2 / (a ** 3))
    m0 = _true_to_mean_anomaly(math.radians(target.true_anomaly_deg), e)
    mean_anomaly = (m0 + n * float(epoch_seconds)) % (2.0 * math.pi)
    eccentric_anomaly = _solve_kepler(mean_anomaly, e)
    x_p = a * (math.cos(eccentric_anomaly) - e)
    y_p = a * math.sqrt(1.0 - e * e) * math.sin(eccentric_anomaly)
    return _rotate_pqw_to_eci(
        x_p,
        y_p,
        math.radians(target.raan_deg),
        math.radians(target.inclination_deg),
        math.radians(target.arg_perigee_deg),
    )


def _true_to_mean_anomaly(true_anomaly_rad: float, eccentricity: float) -> float:
    if eccentricity <= 0.0:
        return true_anomaly_rad % (2.0 * math.pi)
    e_anomaly = 2.0 * math.atan2(
        math.sqrt(1.0 - eccentricity) * math.sin(true_anomaly_rad / 2.0),
        math.sqrt(1.0 + eccentricity) * math.cos(true_anomaly_rad / 2.0),
    )
    return (e_anomaly - eccentricity * math.sin(e_anomaly)) % (2.0 * math.pi)


def _solve_kepler(mean_anomaly: float, eccentricity: float) -> float:
    e_anomaly = mean_anomaly
    for _ in range(12):
        delta = (e_anomaly - eccentricity * math.sin(e_anomaly) - mean_anomaly) / (
            1.0 - eccentricity * math.cos(e_anomaly)
        )
        e_anomaly -= delta
        if abs(delta) < 1e-12:
            break
    return e_anomaly


def _rotate_pqw_to_eci(
    x_p: float,
    y_p: float,
    raan: float,
    inclination: float,
    arg_perigee: float,
) -> tuple[float, float, float]:
    cos_o, sin_o = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(inclination), math.sin(inclination)
    cos_w, sin_w = math.cos(arg_perigee), math.sin(arg_perigee)
    r11 = cos_o * cos_w - sin_o * sin_w * cos_i
    r12 = -cos_o * sin_w - sin_o * cos_w * cos_i
    r21 = sin_o * cos_w + cos_o * sin_w * cos_i
    r22 = -sin_o * sin_w + cos_o * cos_w * cos_i
    r31 = sin_w * sin_i
    r32 = cos_w * sin_i
    return (r11 * x_p + r12 * y_p, r21 * x_p + r22 * y_p, r31 * x_p + r32 * y_p)


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) ** 2 for x in vector))


def _unit(vector: Sequence[float]) -> tuple[float, float, float]:
    norm = _norm(vector)
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return tuple(float(x) / norm for x in vector)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))
