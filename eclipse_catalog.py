"""
EclipseCatalog — query interface over the eclipse list.

This module defines the EclipseCatalog class, which wraps a list of
eclipse dictionaries (loaded from eclipse_data.json) and provides
methods for:

  - Date parsing          (parse_date)
  - Visibility estimation (visibility_radius_km, is_visible_from)
  - Local-view classification with confidence (local_view)
  - Flexible date search  (find_by_date, find_index_by_date)
  - Future-eclipse lookup  (find_next_visible)
  - Text summaries for LLM context (summary)
  - Dropdown labels       (labels property)

It depends on helper functions and the Eclipse class from
eclipse_utils.py but contains no geometry math itself — it is
purely a data-access and classification layer.

Note: local-view classification uses a parallax-calibrated geometric
heuristic model. Results are estimates, especially near path edges.
"""

from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

from eclipse_utils import (
    MONTH_NUM,
    parse_coord,
    haversine_km,
    overlap_fraction,
    Eclipse,
)


# =================================================================
# EclipseCatalog class
# =================================================================

class EclipseCatalog:
    """
    Owns the list of eclipse dicts and provides query methods.

    Usage:
        catalog = EclipseCatalog(data["eclipse_list"])
        catalog.find_by_date("2017 Aug 21")
        catalog.find_next_visible(lat, lon, n=3)
    """

    # ---------------------------------------------------------
    # Construction and container interface
    # ---------------------------------------------------------
    # The catalog wraps a plain list of eclipse dicts so that
    # len(catalog) and catalog[i] work naturally, while also
    # providing domain-specific query methods.
    # ---------------------------------------------------------

    def __init__(self, eclipse_list: List[Dict[str, Any]]):
        self.eclipses = eclipse_list

    def __len__(self) -> int:
        return len(self.eclipses)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.eclipses[index]

    # ---------------------------------------------------------
    # Date parsing
    # ---------------------------------------------------------
    # Converts the catalog's "YYYY Mon DD" date strings into
    # Python datetime objects for comparison and filtering.
    # ---------------------------------------------------------

    @staticmethod
    def parse_date(date_str: str) -> Optional[datetime]:
        """Convert 'YYYY Mon DD' to a datetime, or None on failure."""
        try:
            parts = date_str.strip().split()
            return datetime(int(parts[0]), MONTH_NUM.get(parts[1], 1), int(parts[2]))
        except Exception:
            return None

    # ---------------------------------------------------------
    # Visibility helpers
    # ---------------------------------------------------------
    # These estimate whether an eclipse is at least partially
    # visible from a given latitude/longitude, using the path
    # width and a generous partial-visibility radius (~3500 km
    # beyond the central path edge, or ~2500 km for partials).
    # ---------------------------------------------------------

    @staticmethod
    def visibility_radius_km(eclipse: Dict[str, Any]) -> float:
        """Rough radius (km) within which the eclipse is at least partially visible."""
        raw = eclipse.get("_raw", {})
        pw = raw.get("path_width_km", "-")
        try:
            path_w = float(pw)
        except (ValueError, TypeError):
            path_w = 0
        return (path_w / 2 + 3500) if path_w > 0 else 2500

    def is_visible_from(self, eclipse: Dict[str, Any], lat: float, lon: float) -> bool:
        """True if the eclipse is roughly visible from (lat, lon)."""
        raw = eclipse.get("_raw", {})
        ecl_lat = parse_coord(raw.get("latitude", "0N"))
        ecl_lon = parse_coord(raw.get("longitude", "0E"))
        dist = haversine_km(lat, lon, ecl_lat, ecl_lon)
        return dist <= self.visibility_radius_km(eclipse)

    # ---------------------------------------------------------
    # Local view classification
    # ---------------------------------------------------------
    # Estimates what an observer at (lat, lon) would experience
    # during a given eclipse: likely totality, annularity,
    # partial, or not visible — plus a confidence note.
    #
    # For central eclipses it checks geometric path membership
    # first (more reliable), then falls back to parallax-based
    # obscuration. Results are labeled as estimates because the
    # model is a heuristic approximation, not full Besselian
    # element computation.
    # ---------------------------------------------------------

    @staticmethod
    def local_view(eclipse: Dict[str, Any], lat: float, lon: float) -> Tuple[str, float, str]:
        """
        Classify local eclipse experience at maximum eclipse.

        Returns (label, obscuration_fraction, confidence_note).
        """
        eclipse_obj = Eclipse(eclipse)
        sun_r = 1.0
        moon_r = (eclipse.get("magnitude") or 0.95) * sun_r
        d = abs(eclipse_obj.parallax_offset(lat, lon))
        obsc = overlap_fraction(sun_r, moon_r, d)

        if eclipse_obj.is_central:
            dist_km = haversine_km(lat, lon, eclipse_obj.ge_lat, eclipse_obj.ge_lon)
            eff_half = eclipse_obj.effective_half_width(dist_km)
            perp_km = eclipse_obj.perpendicular_distance_km(lat, lon)
            in_central_path = eff_half > 0 and perp_km <= (1.1 * eff_half)
            edge_ratio = (perp_km / eff_half) if eff_half > 0 else 99.0
            confidence = "High confidence" if edge_ratio < 0.8 else "Near path edge (lower confidence)"
            if in_central_path:
                if moon_r > sun_r:
                    return "Likely totality", max(obsc, 0.95), confidence
                if moon_r < sun_r:
                    return "Likely annularity", max(obsc, 0.90), confidence

        if obsc <= 0:
            return "Likely not visible", 0.0, "Approximate model"
        return "Likely partial", obsc, "Approximate model"

    # ---------------------------------------------------------
    # Search methods
    # ---------------------------------------------------------
    # Flexible text matching for eclipse dates. Supports full
    # dates ("2017 Aug 21"), partial dates ("2017"), and
    # mixed-format queries ("August 21, 2017"). Also provides
    # index-based lookup for syncing with the Streamlit viewer.
    # ---------------------------------------------------------

    def find_by_date(self, target_date_str: str) -> List[Dict[str, Any]]:
        """Find eclipses matching a date string (flexible matching)."""
        target = target_date_str.strip().lower()
        matches = []
        for ecl in self.eclipses:
            raw_date = ecl["date_raw"].lower()
            if target in raw_date or raw_date in target:
                matches.append(ecl)
        if not matches:
            for ecl in self.eclipses:
                raw_date = ecl["date_raw"].lower()
                tokens = target.replace(",", " ").split()
                if all(t in raw_date for t in tokens):
                    matches.append(ecl)
        return matches

    def find_index_by_date(self, date_str: str) -> Optional[int]:
        """Find the index of the first eclipse matching a date string."""
        target = date_str.strip().lower()
        for i, ecl in enumerate(self.eclipses):
            raw = ecl["date_raw"].lower()
            if target in raw or raw in target:
                return i
        tokens = target.replace(",", " ").split()
        for i, ecl in enumerate(self.eclipses):
            raw = ecl["date_raw"].lower()
            if all(t in raw for t in tokens):
                return i
        return None

    def find_next_visible(self, lat: float, lon: float, n: int = 3,
                          after_date: Optional[datetime] = None) -> List[Tuple[Dict[str, Any], datetime, float]]:
        """
        Find the next n eclipses visible from (lat, lon) after a given date.

        Returns list of (eclipse_dict, datetime, distance_km).
        """
        if after_date is None:
            after_date = datetime.now()
        results: List[Tuple[Dict[str, Any], datetime, float]] = []
        for ecl in self.eclipses:
            dt = self.parse_date(ecl["date_raw"])
            if dt is None or dt < after_date:
                continue
            if self.is_visible_from(ecl, lat, lon):
                raw = ecl.get("_raw", {})
                ecl_lat = parse_coord(raw.get("latitude", "0N"))
                ecl_lon = parse_coord(raw.get("longitude", "0E"))
                dist = haversine_km(lat, lon, ecl_lat, ecl_lon)
                results.append((ecl, dt, dist))
            if len(results) >= n:
                break
        return results

    # ---------------------------------------------------------
    # Summary and display helpers
    # ---------------------------------------------------------
    # summary() builds a multi-line text block injected into
    # the LLM prompt as eclipse context. If observer coords are
    # provided, it appends local-view classification and
    # confidence. The labels property gives one-line strings
    # suitable for dropdown menus in the Streamlit UI.
    # ---------------------------------------------------------

    def summary(self, ecl: Dict[str, Any], obs_lat: Optional[float] = None,
                obs_lon: Optional[float] = None) -> str:
        """Build a text summary of an eclipse, optionally for a specific observer."""
        raw = ecl.get("_raw", {})
        ecl_lat = parse_coord(raw.get("latitude", "0N"))
        ecl_lon = parse_coord(raw.get("longitude", "0E"))
        lines = [
            f"Date: {ecl['date_raw']}",
            f"Type: {ecl['type']} (code: {ecl.get('type_code', '?')})",
            f"Magnitude: {ecl.get('magnitude', '?')}",
            f"Saros: {ecl.get('saros', '?')}",
            f"Duration: {ecl.get('duration', 'N/A')}",
            f"Greatest Eclipse at: {ecl_lat:.1f}°N, {ecl_lon:.1f}°E",
            f"Path Width: {raw.get('path_width_km', 'N/A')} km",
            f"Gamma: {raw.get('gamma', '?')}",
        ]
        if obs_lat is not None and obs_lon is not None:
            dist = haversine_km(obs_lat, obs_lon, ecl_lat, ecl_lon)
            local_label, local_obsc, confidence = self.local_view(ecl, obs_lat, obs_lon)
            lines.append(f"Observer distance: {dist:,.0f} km from center")
            lines.append(f"Estimated local view at max eclipse: {local_label}")
            lines.append(f"Estimated max local obscuration: {local_obsc:.1%}")
            lines.append(f"Model confidence: {confidence}")
        return "\n".join(lines)

    @property
    def labels(self) -> List[str]:
        """One-line label per eclipse, useful for dropdown menus."""
        return [
            f"{e['date_raw']}  —  {e['type']}  (Mag: {e['magnitude']})"
            for e in self.eclipses
        ]
