"""
Shared utilities for eclipse calculations and data handling.
Contains the Eclipse class and helper functions used across notebooks and the Streamlit app.
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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points on Earth."""
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


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


# Backward-compatible wrapper used by draw_eclipse_sky
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
