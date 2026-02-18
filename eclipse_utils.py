"""
Shared utilities for eclipse calculations and data handling.

This module provides the core astronomy and geometry functions used
across the project. It includes:

  - Coordinate parsing (NASA format → signed degrees)
  - Haversine formula (great-circle distance on a sphere)
  - Circle-overlap obscuration (how much of the Sun the Moon covers)
  - The Eclipse class (parallax-calibrated geometric heuristic model
    for estimating what an observer sees during an eclipse)
  - Helper wrappers for the Streamlit app

The Eclipse class uses a simplified geometric model — not full
Besselian element computation — so results are estimates, especially
near path boundaries.
"""

import re
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Month name to number mapping
MONTH_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


# =================================================================
# Coordinate parsing
# =================================================================
# NASA's eclipse catalog encodes latitude/longitude as a number
# followed by a cardinal direction letter: "11S" = 11° South,
# "131W" = 131° West. This function converts that to signed
# decimal degrees (negative for South and West) so the rest of
# the codebase can do normal arithmetic with coordinates.
# =================================================================

def parse_coord(coord_str: str) -> float:
    """Parse NASA coordinate string like '11S' or '131W' to signed degrees."""
    if not coord_str or coord_str.strip() == "-":
        return 0.0
    match = re.match(r"(\d+)([NSEW])", coord_str.strip())
    if match:
        val = float(match.group(1))
        if match.group(2) in ("S", "W"):
            val = -val
        return val
    return 0.0


# =================================================================
# Haversine formula
# =================================================================
# Computes the great-circle (shortest surface) distance between
# two points on a sphere. The formula uses the law of haversines:
#
#   a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
#   d = 2R · arcsin(√a)
#
# where R = 6371 km (mean Earth radius). This is the standard
# method for computing distances on a globe when you only have
# latitude and longitude. It assumes a perfect sphere, which
# introduces < 0.3% error compared to the WGS-84 ellipsoid.
# =================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points on Earth."""
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# =================================================================
# Circle-overlap obscuration
# =================================================================
# During a solar eclipse, the Sun and Moon appear as two circular
# disks in the sky. This function computes the fraction of the
# Sun's area that is covered (obscured) by the Moon, given:
#
#   sun_r  — apparent radius of the Sun  (arbitrary units)
#   moon_r — apparent radius of the Moon (same units)
#   d      — center-to-center distance between the two disks
#
# Three geometric cases:
#   1. d >= sun_r + moon_r  → no overlap (0%)
#   2. d <= |sun_r - moon_r| → full containment (100% if Moon
#      is larger; else (moon_r/sun_r)² for annular geometry)
#   3. Otherwise → partial overlap computed as the area of the
#      lens-shaped intersection of two circles, using the
#      standard two-circle intersection formula.
#
# This is a purely geometric calculation — it does not account
# for limb darkening or atmospheric refraction.
# =================================================================

def overlap_fraction(sun_r: float, moon_r: float, d: float) -> float:
    """
    Fraction of the Sun's disk area obscured by the Moon.

    Parameters
    ----------
    sun_r, moon_r : float
        Apparent radii (same units)
    d : float
        Center-to-center distance

    Returns
    -------
    float
        Fraction of Sun's disk obscured (0.0 to 1.0)
    """
    if d >= sun_r + moon_r:
        return 0.0
    if d <= abs(sun_r - moon_r):
        return 1.0 if moon_r >= sun_r else (moon_r / sun_r) ** 2

    # Partial overlap — lens-shaped intersection of two circles
    cos1 = np.clip((d**2 + sun_r**2 - moon_r**2) / (2 * d * sun_r), -1, 1)
    cos2 = np.clip((d**2 + moon_r**2 - sun_r**2) / (2 * d * moon_r), -1, 1)
    a1 = sun_r**2 * np.arccos(cos1)
    a2 = moon_r**2 * np.arccos(cos2)
    det = (-d + sun_r + moon_r) * (d + sun_r - moon_r) * \
          (d - sun_r + moon_r) * (d + sun_r + moon_r)
    a3 = 0.5 * np.sqrt(max(det, 0))
    return (a1 + a2 - a3) / (np.pi * sun_r**2)


# =================================================================
# Eclipse class
# =================================================================
# Models a single solar eclipse and estimates what an observer at
# any location on Earth would see, using a parallax-calibrated
# geometric heuristic.
#
# Key astronomy concepts used:
#
# • Magnitude — ratio of the Moon's apparent diameter to the
#   Sun's. Mag > 1 means the Moon fully covers the Sun (total);
#   mag < 1 means a ring of Sun remains visible (annular).
#
# • Gamma — how close the Moon's shadow axis passes to Earth's
#   center, measured in Earth radii. |γ| < 1 means the shadow
#   touches the surface; higher values give partial-only eclipses.
#
# • Saros cycle — eclipses repeat every ~18 years 11 days in
#   the same Saros series, with similar geometry each time.
#
# • Path width — the width of the Moon's umbral (or antumbral)
#   shadow on Earth's surface. Only observers inside this narrow
#   corridor see totality (or annularity).
#
# • Topocentric parallax — the Moon is close enough (~384 400 km)
#   that an observer's position on Earth's surface shifts where
#   the Moon appears against the Sun. This is why the path of
#   totality is narrow: a few hundred km sideways changes the
#   Moon's apparent position enough to break total coverage.
#
# The class uses the catalog's path width to *calibrate* this
# parallax effect: at the edge of the totality path, the Moon's
# offset equals exactly (magnitude − 1) Sun-radii — the
# threshold where totality ends. Farther away, the offset grows
# proportionally, giving a smooth partial-eclipse falloff.
#
# Limitations:
#   - Uses one reference point (greatest eclipse), not the full
#     centerline, so accuracy degrades far from that point.
#   - Path bearing is estimated from the season, not computed
#     from orbital elements.
#   - Shadow width tapering is a linear heuristic.
#   - Not suitable for precise contact-time predictions.
# =================================================================

class Eclipse:
    """
    Represents a single solar eclipse with methods for observer
    visibility calculations using parallax-based geometry.

    The Moon's topocentric parallax (~0.95°) shifts its apparent
    position for observers displaced from the shadow centerline.
    The known path width calibrates this angular conversion so that
    the edge of totality maps exactly to the Moon-Sun radius
    difference (magnitude − 1).
    """

    # Average Earth-Moon distance (km) — used for parallax
    MOON_DISTANCE_KM = 384_400
    # Sun angular radius (radians, ~16 arcmin)
    SUN_ANGULAR_RAD = 0.00465

    def __init__(self, eclipse_dict: Dict[str, Any]):
        self.data = eclipse_dict
        self._raw = eclipse_dict.get("_raw", {})

        # Core properties
        self.date_raw = eclipse_dict.get("date_raw", "")
        self.year = eclipse_dict.get("year", 0)
        self.type_code = eclipse_dict.get("type_code", "")
        self.type_name = eclipse_dict.get("type", "")
        self.magnitude = eclipse_dict.get("magnitude") or 0.95
        self.saros = eclipse_dict.get("saros", "")

        # Greatest-eclipse coordinates
        self.ge_lat = parse_coord(self._raw.get("latitude", "0N"))
        self.ge_lon = parse_coord(self._raw.get("longitude", "0E"))

        # Path width (km) — 0 for partial eclipses
        pw = self._raw.get("path_width_km", "-")
        try:
            self.path_width_km = float(pw)
        except (ValueError, TypeError):
            self.path_width_km = 0.0

        # Gamma (shadow-axis distance from Earth center, in Earth radii)
        try:
            self.gamma = float(self._raw.get("gamma", "0"))
        except (ValueError, TypeError):
            self.gamma = 0.0

    def __repr__(self) -> str:
        return (f"Eclipse({self.date_raw}, {self.type_name}, "
                f"mag={self.magnitude})")

    @property
    def is_central(self) -> bool:
        """True for Total, Annular, or Hybrid eclipses."""
        return self.path_width_km > 0

    @property
    def path_half_width(self) -> float:
        """Half the path width at greatest eclipse (km)."""
        return self.path_width_km / 2.0

    def describe(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            f"Date:       {self.date_raw}",
            f"Type:       {self.type_name}  (magnitude {self.magnitude})",
            f"GE point:   {self.ge_lat:.1f}°N, {self.ge_lon:.1f}°E",
            f"Gamma:      {self.gamma}",
            f"Saros:      {self.saros}",
        ]
        if self.is_central:
            lines.append(f"Path width: {self.path_width_km:.0f} km")
        return "\n".join(lines)

    # --- Path bearing estimation ---
    # The Moon's shadow always sweeps west-to-east because the
    # Moon orbits eastward. But the path tilts north or south
    # depending on the season: in spring the Sun moves north in
    # declination, pulling the shadow path northeast; in fall it
    # moves south, giving a southeast path. Near a solstice the
    # declination is nearly stationary, so the path runs roughly
    # due east. This method approximates the tilt angle from the
    # day-of-year using a cosine model of the Sun's declination
    # rate of change.

    def estimate_path_bearing(self) -> float:
        """
        Estimate the eclipse-path bearing from the date.

        The Moon's shadow sweeps west → east. Whether the path
        tilts NE or SE depends on the rate of change of the Sun's
        declination:
          • Spring (Sun moving north) → NE path  (~40°)
          • Fall   (Sun moving south) → SE path  (~140°)
          • Near solstice             → ~due east (~90°)
        """
        parts = self.date_raw.split()
        month = MONTH_NUM.get(parts[1], 6) if len(parts) > 1 else 6
        day = int(parts[2]) if len(parts) > 2 else 15
        doy = (month - 1) * 30.4 + day

        dec_rate = np.cos(2 * np.pi * (doy - 80) / 365.25)
        return 90.0 - dec_rate * 50.0

    # --- Shadow width tapering ---
    # The umbral shadow is widest at the point of greatest
    # eclipse and narrows as it moves along the path. This is
    # because Earth's curvature increases the slant distance to
    # the shadow cone tip, making the cone cross-section smaller
    # on the surface. This method applies a simple linear taper:
    # at 3000 km from greatest eclipse the width is reduced by
    # up to 60%, with a floor at 50% of the original width.

    def effective_half_width(self, dist_from_ge_km: float) -> float:
        """
        Path half-width adjusted for distance from greatest eclipse.
        The umbral shadow narrows as it moves away from the point of
        greatest eclipse because the shadow cone diverges and the
        Earth's curvature increases the slant distance.
        """
        if not self.is_central:
            return 0.0
        factor = max(0.5, 1.0 - dist_from_ge_km / 3000.0 * 0.6)
        return self.path_half_width * factor

    # --- Cross-track (perpendicular) distance ---
    # To decide if an observer is inside the totality corridor,
    # we need their perpendicular distance from the centerline.
    # The centerline direction is estimated from the path bearing.
    # Then the spherical cross-track formula gives the shortest
    # distance from the observer to that great-circle line:
    #
    #   cross_track = R · arcsin(sin(d13) · sin(bearing_diff))
    #
    # where d13 is the angular distance from the greatest-eclipse
    # point to the observer. A gamma-based latitude gate quickly
    # rejects observers too far north or south to possibly be in
    # the path, saving computation.

    def perpendicular_distance_km(self, obs_lat: float, obs_lon: float) -> float:
        """
        Estimate the observer's perpendicular distance (km) from the
        eclipse centerline using cross-track geometry.

        Steps:
        1. Gamma-based latitude gate  (high |γ| → narrow band)
        2. Bearing-constrained search (±2° around estimated bearing)
        3. Spherical cross-track formula
        """
        R = 6371.0
        d13_km = haversine_km(obs_lat, obs_lon, self.ge_lat, self.ge_lon)

        if d13_km < 200:
            return d13_km  # close enough — direct distance OK

        # Latitude gate
        ga = min(abs(self.gamma), 0.999)
        lat_range = np.degrees(np.arccos(ga)) * 0.55 + 5.0
        if abs(obs_lat - self.ge_lat) > lat_range:
            return d13_km

        d13 = d13_km / R
        lat1, lon1 = np.radians(self.ge_lat), np.radians(self.ge_lon)
        lat2, lon2 = np.radians(obs_lat), np.radians(obs_lon)
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = (np.cos(lat1) * np.sin(lat2)
             - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
        brg_obs = np.arctan2(x, y)

        bearing = self.estimate_path_bearing()
        min_xt = d13_km

        for b_off in range(-2, 3):
            brg_path = np.radians(bearing + b_off)
            xt = abs(R * np.arcsin(
                np.clip(np.sin(d13) * np.sin(brg_obs - brg_path), -1, 1)
            ))
            if xt < min_xt:
                cos_xt = np.cos(xt / R)
                if cos_xt > 1e-10:
                    cos_at = np.clip(np.cos(d13) / cos_xt, -1, 1)
                    at = R * np.arccos(cos_at)
                    if at <= 7500:
                        min_xt = xt
        return min_xt

    # --- Parallax-calibrated offset ---
    # This is the central astronomy function. It estimates how
    # far the Moon appears to be displaced from the Sun's center
    # for a specific observer, expressed in Sun-radii.
    #
    # Physical basis: the Moon is only ~384 400 km away, so
    # moving a few hundred km sideways on Earth shifts the
    # Moon's apparent position against the distant Sun. The
    # catalog's path width tells us exactly how far sideways
    # you can be before totality ends. We use that as a
    # calibration anchor:
    #
    #   At the path edge: offset = (magnitude − 1) × sun_radius
    #
    # Inside the path the offset is smaller (totality), outside
    # it grows (partial eclipse), and far enough away the Moon
    # misses the Sun entirely (no eclipse).

    def parallax_offset(self, obs_lat: float, obs_lon: float) -> float:
        """
        Moon's apparent offset from the Sun's center for this observer,
        in Sun-radii units.

        Physics:
        The Moon is ~384 400 km away, so an observer displaced from
        the shadow centerline sees the Moon shifted by the topocentric
        parallax. The known path width calibrates the conversion:

            At perpendicular distance = effective_half_width :
                offset = (magnitude − 1) × sun_radius

        which is *exactly* the threshold where the Moon just barely
        covers the Sun (edge of totality).

        Returns
        -------
        offset : float
            0       → Moon centered on Sun
            < mag−1 → totality (Moon fully covers Sun)
            > mag−1 → partial (some Sun visible)
            > 1+mag → no eclipse visible
        """
        sun_r = 1.0
        moon_r = self.magnitude * sun_r
        max_off = sun_r + moon_r

        dist_km = haversine_km(obs_lat, obs_lon, self.ge_lat, self.ge_lon)

        if self.is_central:
            eff_half = self.effective_half_width(dist_km)
            perp_km = self.perpendicular_distance_km(obs_lat, obs_lon)

            # Parallax-calibrated offset:
            #   offset / (mag − 1) = perp_km / eff_half
            # So at the path edge offset = (mag−1) exactly.
            if eff_half > 0:
                offset = perp_km * (self.magnitude - 1) * sun_r / eff_half
            else:
                offset = dist_km / 3500.0 * max_off

            return min(offset, max_off + 0.3)
        else:
            # Partial eclipse — gamma-based base offset
            base = abs(self.gamma) * 0.6
            return min(base + dist_km / 3500.0 * 0.6, max_off + 0.3)


# =================================================================
# Convenience wrappers
# =================================================================
# These thin wrappers let the Streamlit app call Eclipse methods
# without constructing the object manually every time.
# =================================================================

def viewer_offset(obs_lat: float, obs_lon: float, eclipse: Dict[str, Any]) -> float:
    """Wrapper: create Eclipse object and call parallax_offset."""
    return Eclipse(eclipse).parallax_offset(obs_lat, obs_lon)


def get_eclipse_params(eclipse: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    """Unpack useful numbers from an eclipse dict."""
    raw = eclipse.get("_raw", {})
    ecl_lat = parse_coord(raw.get("latitude", "0N"))
    ecl_lon = parse_coord(raw.get("longitude", "0E"))
    magnitude = eclipse.get("magnitude") or 0.95
    pw = raw.get("path_width_km", "-")
    try:
        path_w = float(pw)
    except (ValueError, TypeError):
        path_w = 0.0
    gamma_str = raw.get("gamma", "0")
    try:
        gamma = float(gamma_str)
    except (ValueError, TypeError):
        gamma = 0.0
    return ecl_lat, ecl_lon, magnitude, path_w, gamma
