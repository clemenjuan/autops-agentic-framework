"""Physical constants.

Some of these also exist inside Orekit, but OreKit isolation forbids importing
Orekit under adcs/, so they are re-declared here. Duplication is
is checked by test_constants_match_orekit.
"""

import numpy as np

### Earth shape and gravity ###
# WGS84 defines the equatorial radius exactly.
R_EARTH = 6378137.0          # m, equatorial radius (NIMA TR8350.2, WGS84)
MU_EARTH = 3.986004418e14    # m^3/s^2, gravitational parameter GM (NIMA TR8350.2)
OMEGA_EARTH = 7.2921159e-5   # rad/s, sidereal rotation rate (NIMA TR8350.2)

### Solar radiation ###
# SOLAR_PRESSURE and SOLAR_CONSTANT are deliberately NOT consistent with each
# other. SOLAR_PRESSURE is Paluszek's value, derived from the older 1367 W/m^2
# figure, and is kept as a literal so the SRP torque is unchanged from when
# dynamics.py was written and tested. SOLAR_CONSTANT is the modern measured
# value, used where an irradiance is needed directly. The two differ by 0.45 %.
# Reconciling them would change the SRP torque, and no current test would
# catch it - test_srp_eclipse_gate only checks nonzero versus zero.
SOLAR_PRESSURE = 4.56e-6     # N/m^2 at 1 AU (Paluszek Table 8.1; 1367 W/m^2 / c)
SOLAR_CONSTANT = 1360.8      # W/m^2 at 1 AU (Kopp & Lean 2011, GRL 38, L01706)


### Earth reflectivity ###
# Bond albedo: the fraction of incident sunlight reflected, summed over all
# wavelengths and directions.
EARTH_BOND_ALBEDO = 0.306    # NASA Earth Fact Sheet (Williams, D. R.)