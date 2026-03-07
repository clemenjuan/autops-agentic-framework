"""Orbital mechanics module.

Provides high-fidelity orbital propagation, eclipse computation, and
ground station visibility via Orekit (optional dependency). Falls back
to simplified analytical models when Orekit is not available.
"""

from src.environment.orbital.context import OrbitalContext, compute_orbital_context

__all__ = ["OrbitalContext", "compute_orbital_context"]
