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

# ============================================================
# Imports
# ============================================================
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

# Import geometry utilities and Eclipse class from eclipse_utils
from eclipse_utils import (
    MONTH_NUM,          # Month name to number mapping (e.g., "Jan" → 1)
    parse_coord,         # Convert NASA coordinate format ("11S" → -11.0)
    haversine_km,        # Great-circle distance between two points on Earth
    overlap_fraction,    # Calculate Sun/Moon disk overlap obscuration
    Eclipse,             # Eclipse class for geometric calculations
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
        """
        Initialize catalog with a list of eclipse dictionaries.
        
        Parameters:
        -----------
        eclipse_list : List[Dict[str, Any]]
            List of eclipse dictionaries loaded from eclipse_data.json
        """
        self.eclipses = eclipse_list

    def __len__(self) -> int:
        """Return the number of eclipses in the catalog."""
        return len(self.eclipses)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get eclipse at index (enables catalog[i] syntax)."""
        return self.eclipses[index]

    # ---------------------------------------------------------
    # Date parsing
    # ---------------------------------------------------------
    # Converts the catalog's "YYYY Mon DD" date strings into
    # Python datetime objects for comparison and filtering.
    # ---------------------------------------------------------

    @staticmethod
    def parse_date(date_str: str) -> Optional[datetime]:
        """
        Convert 'YYYY Mon DD' date string to a datetime object.
        
        Used for date comparisons and filtering (e.g., finding future eclipses).
        Returns None if the date string cannot be parsed.
        
        Example:
            parse_date("2024 Apr 08") → datetime(2024, 4, 8)
        """
        try:
            parts = date_str.strip().split()
            # parts[0] = year, parts[1] = month abbreviation, parts[2] = day
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
        """
        Estimate the radius (km) within which the eclipse is at least partially visible.
        
        For central eclipses (Total/Annular/Hybrid), uses path width + 3500 km buffer.
        For partial eclipses, uses a default 2500 km radius.
        This is a generous estimate to ensure we don't miss visible eclipses.
        """
        raw = eclipse.get("_raw", {})
        pw = raw.get("path_width_km", "-")
        try:
            path_w = float(pw)
        except (ValueError, TypeError):
            path_w = 0
        
        # Central eclipses: half path width + 3500 km buffer for partial visibility
        # Partial eclipses: default 2500 km radius
        return (path_w / 2 + 3500) if path_w > 0 else 2500

    def is_visible_from(self, eclipse: Dict[str, Any], lat: float, lon: float) -> bool:
        """
        Check if an eclipse is roughly visible from a given location.
        
        Returns True if the observer is within the visibility radius of the
        eclipse center. This is a quick filter for finding relevant eclipses.
        """
        raw = eclipse.get("_raw", {})
        # Get eclipse center coordinates
        ecl_lat = parse_coord(raw.get("latitude", "0N"))
        ecl_lon = parse_coord(raw.get("longitude", "0E"))
        
        # Calculate distance from observer to eclipse center
        dist = haversine_km(lat, lon, ecl_lat, ecl_lon)
        
        # Check if within visibility radius
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
        Classify what an observer at (lat, lon) would experience during the eclipse.
        
        This estimates whether the observer would see totality, annularity, partial,
        or nothing, along with the obscuration fraction and a confidence note.
        
        For central eclipses, uses geometric path membership (more reliable).
        For all eclipses, also calculates parallax-based obscuration as a fallback.
        
        Returns:
        --------
        tuple: (label, obscuration_fraction, confidence_note)
            label: "Likely totality", "Likely annularity", "Likely partial", or "Likely not visible"
            obscuration_fraction: 0.0 to 1.0 (fraction of Sun covered)
            confidence_note: "High confidence", "Near path edge (lower confidence)", or "Approximate model"
        """
        # Create Eclipse object for geometric calculations
        eclipse_obj = Eclipse(eclipse)
        
        # Set up Sun and Moon sizes (Sun radius = 1.0)
        sun_r = 1.0
        moon_r = (eclipse.get("magnitude") or 0.95) * sun_r
        
        # Calculate parallax-corrected Moon offset from Sun center
        d = abs(eclipse_obj.parallax_offset(lat, lon))
        
        # Calculate obscuration fraction (how much of Sun is covered)
        obsc = overlap_fraction(sun_r, moon_r, d)

        # For central eclipses (Total/Annular/Hybrid), check geometric path membership
        if eclipse_obj.is_central:
            # Distance from observer to eclipse center
            dist_km = haversine_km(lat, lon, eclipse_obj.ge_lat, eclipse_obj.ge_lon)
            
            # Effective half-width of path at observer's distance
            eff_half = eclipse_obj.effective_half_width(dist_km)
            
            # Perpendicular distance from observer to central path
            perp_km = eclipse_obj.perpendicular_distance_km(lat, lon)
            
            # Check if observer is within central path (with 10% buffer)
            in_central_path = eff_half > 0 and perp_km <= (1.1 * eff_half)
            
            # Calculate edge ratio for confidence assessment
            edge_ratio = (perp_km / eff_half) if eff_half > 0 else 99.0
            confidence = "High confidence" if edge_ratio < 0.8 else "Near path edge (lower confidence)"
            
            # If in central path, classify as totality or annularity
            if in_central_path:
                if moon_r > sun_r:  # Moon larger than Sun → totality
                    return "Likely totality", max(obsc, 0.95), confidence
                if moon_r < sun_r:  # Moon smaller than Sun → annularity
                    return "Likely annularity", max(obsc, 0.90), confidence

        # Fallback: use obscuration fraction for classification
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
        """
        Find all eclipses matching a date string (flexible text matching).
        
        Supports various date formats:
        - Full dates: "2017 Aug 21", "August 21, 2017"
        - Partial dates: "2017", "Aug 21"
        - Year only: "2024"
        
        Uses substring matching first, then token-based matching if no results.
        
        Returns:
        --------
        List[Dict[str, Any]]: List of matching eclipse dictionaries
        """
        target = target_date_str.strip().lower()
        matches = []
        
        # First pass: substring matching (e.g., "2017" in "2017 Aug 21")
        for ecl in self.eclipses:
            raw_date = ecl["date_raw"].lower()
            if target in raw_date or raw_date in target:
                matches.append(ecl)
        
        # Second pass: token-based matching (all tokens must be present)
        # Handles cases like "August 21, 2017" → matches "2017 Aug 21"
        if not matches:
            for ecl in self.eclipses:
                raw_date = ecl["date_raw"].lower()
                tokens = target.replace(",", " ").split()
                if all(t in raw_date for t in tokens):
                    matches.append(ecl)
        
        return matches

    def find_index_by_date(self, date_str: str) -> Optional[int]:
        """
        Find the catalog index of the first eclipse matching a date string.
        
        Used by the Streamlit app to sync the viewer with a specific eclipse
        when the chatbot recommends one.
        
        Returns:
        --------
        Optional[int]: Index of matching eclipse, or None if not found
        """
        target = date_str.strip().lower()
        
        # First pass: substring matching
        for i, ecl in enumerate(self.eclipses):
            raw = ecl["date_raw"].lower()
            if target in raw or raw in target:
                return i
        
        # Second pass: token-based matching
        tokens = target.replace(",", " ").split()
        for i, ecl in enumerate(self.eclipses):
            raw = ecl["date_raw"].lower()
            if all(t in raw for t in tokens):
                return i
        
        return None

    def find_next_visible(self, lat: float, lon: float, n: int = 3,
                          after_date: Optional[datetime] = None) -> List[Tuple[Dict[str, Any], datetime, float]]:
        """
        Find the next n eclipses visible from a given location.
        
        Iterates through the catalog (which is sorted chronologically) and
        returns the first n eclipses that:
        1. Occur after the specified date (defaults to now)
        2. Are at least partially visible from (lat, lon)
        
        Used by the chatbot to answer "next eclipse" queries.
        
        Parameters:
        -----------
        lat, lon : float
            Observer's latitude and longitude (degrees)
        n : int
            Number of eclipses to return (default: 3)
        after_date : Optional[datetime]
            Only return eclipses after this date (defaults to now)
        
        Returns:
        --------
        List[Tuple[Dict[str, Any], datetime, float]]:
            List of (eclipse_dict, datetime, distance_km) tuples, sorted by date
        """
        if after_date is None:
            after_date = datetime.now()
        
        results: List[Tuple[Dict[str, Any], datetime, float]] = []
        
        # Iterate through catalog (already sorted chronologically)
        for ecl in self.eclipses:
            # Parse eclipse date
            dt = self.parse_date(ecl["date_raw"])
            if dt is None or dt < after_date:
                continue  # Skip past eclipses or invalid dates
            
            # Check if visible from observer location
            if self.is_visible_from(ecl, lat, lon):
                # Get eclipse center coordinates
                raw = ecl.get("_raw", {})
                ecl_lat = parse_coord(raw.get("latitude", "0N"))
                ecl_lon = parse_coord(raw.get("longitude", "0E"))
                
                # Calculate distance from observer to eclipse center
                dist = haversine_km(lat, lon, ecl_lat, ecl_lon)
                
                results.append((ecl, dt, dist))
            
            # Stop once we have enough results
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
        """
        Build a text summary of an eclipse for LLM context injection.
        
        Creates a multi-line text block with eclipse details. If observer
        coordinates are provided, also includes local-view classification
        and obscuration estimates.
        
        This summary is injected into the LLM prompt as part of the RAG
        (Retrieval-Augmented Generation) pattern.
        
        Parameters:
        -----------
        ecl : Dict[str, Any]
            Eclipse dictionary from the catalog
        obs_lat, obs_lon : Optional[float]
            Observer's coordinates (if provided, adds local-view info)
        
        Returns:
        --------
        str: Multi-line text summary
        """
        raw = ecl.get("_raw", {})
        ecl_lat = parse_coord(raw.get("latitude", "0N"))
        ecl_lon = parse_coord(raw.get("longitude", "0E"))
        
        # Build base summary with eclipse details
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
        
        # Add observer-specific information if coordinates provided
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
        """
        Generate one-line labels for each eclipse (for dropdown menus).
        
        Format: "YYYY Mon DD  —  Type  (Mag: X.XXXX)"
        Used by the Streamlit app's eclipse selector dropdown.
        
        Returns:
        --------
        List[str]: List of formatted eclipse labels
        """
        return [
            f"{e['date_raw']}  —  {e['type']}  (Mag: {e['magnitude']})"
            for e in self.eclipses
        ]
