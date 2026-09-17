"""Mission architecture subpackage.

Scenario-agnostic mission interface (`adcs_mission_base`) plus the concrete
missions built on it. A mission owns the pointing target sequence and the
per-step success/failure bookkeeping; the ADCS simulation stays unaware of
what the spacecraft is trying to achieve.
"""
