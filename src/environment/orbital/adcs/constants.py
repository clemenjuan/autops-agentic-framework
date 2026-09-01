"""Physical constants.

Some of these also exist inside Orekit, but OreKit isolation forbids importing
Orekit under adcs/, so they are re-declared here. Duplication is
is checked by test_constants_match_orekit.
"""

### Earth shape and gravity ###
# WGS84 defines the equatorial radius exactly.
R_EARTH = 6378137.0          # m, equatorial radius (NIMA TR8350.2, WGS84)
MU_EARTH = 3.986004418e14    # m^3/s^2, gravitational parameter GM (NIMA TR8350.2)
OMEGA_EARTH = 7.292115e-5   # rad/s, sidereal rotation rate (NIMA TR8350.2)

### Solar radiation ###
SOLAR_PRESSURE = 4.56e-6     # N/m^2 at 1 AU (Paluszek Table 8.1; 1367 W/m^2 / c)

### Earth reflectivity ###
# Bond albedo: the fraction of incident sunlight reflected, summed over all
# wavelengths and directions.
EARTH_BOND_ALBEDO = 0.306    # NASA Earth Fact Sheet (Williams, D. R.)