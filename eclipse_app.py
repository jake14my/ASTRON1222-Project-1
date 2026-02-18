"""
Eclipse Explorer — Streamlit App
Combines the Eclipse Chatbot (LLM-powered) with the Eclipse Viewer visualization.
The chatbot can set the latitude, longitude, and eclipse for the viewer automatically.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import litellm
from litellm import completion
import streamlit as st
from eclipse_utils import (
    parse_coord,
    haversine_km,
    overlap_fraction,
    get_eclipse_params,
    Eclipse,
    viewer_offset as eclipse_viewer_offset,
)
from eclipse_catalog import EclipseCatalog

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Eclipse Explorer", page_icon="🌒", layout="wide")

# ============================================================
# LOAD ENV & DATA
# ============================================================
load_dotenv()
api_key = os.environ.get("ASTRO1221_API_KEY")
API_BASE = "https://litellmproxy.osu-ai.org"

@st.cache_data
def load_eclipse_data():
    try:
        with open("eclipse_data.json") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Error: eclipse_data.json not found. Please run EclipseData.ipynb first to generate the data file.")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Error: eclipse_data.json is corrupted or invalid. {e}")
        st.stop()

data = load_eclipse_data()
eclipse_list = data["eclipse_list"]

# Validate eclipse_list is not empty
if not eclipse_list:
    st.error("Error: No eclipse data found in eclipse_data.json")
    st.stop()

catalog = EclipseCatalog(eclipse_list)

# ============================================================
# GEOMETRY HELPERS (using shared utilities from eclipse_utils)
# Additional functions specific to Streamlit app below
# ============================================================


def viewer_offset(obs_lat, obs_lon, eclipse):
    """Parallax-calibrated Moon offset from the Sun center (Sun-radii units).
    Uses the Eclipse class from eclipse_utils for consistent calculations."""
    return eclipse_viewer_offset(obs_lat, obs_lon, eclipse)


# Visibility, local-view, and search methods are now on the
# EclipseCatalog instance (`catalog`) imported from eclipse_utils.


# ============================================================
# DRAWING FUNCTION
# ============================================================

def draw_eclipse_sky(eclipse, obs_lat, obs_lon, time_frac):
    ecl_lat, ecl_lon, mag, path_w, _ = get_eclipse_params(eclipse)
    sun_r = 1.0
    moon_r = mag * sun_r
    offset = viewer_offset(obs_lat, obs_lon, eclipse)

    travel = sun_r + moon_r + 0.5
    mx = -time_frac * travel
    my = offset

    d = np.sqrt(mx**2 + my**2)
    obs_frac = overlap_fraction(sun_r, moon_r, d)
    dist_km = haversine_km(obs_lat, obs_lon, ecl_lat, ecl_lon)
    ecl_type = eclipse.get("type", "?")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")

    bright = max(0.02, 0.12 - 0.10 * obs_frac)
    ax.set_facecolor((bright, bright, bright + 0.12))

    if obs_frac > 0.90:
        alpha_base = (obs_frac - 0.90) / 0.10 * 0.45
        for i in range(10):
            r = sun_r + 0.05 * (i + 1)
            a = alpha_base * (1 - i / 10)
            ax.add_patch(plt.Circle((0, 0), r, color="#FFE4B5", alpha=max(a, 0), lw=0))

    ax.add_patch(plt.Circle((0, 0), sun_r, color="#FFD700", zorder=1))
    for i in range(4):
        ax.add_patch(plt.Circle((0, 0), sun_r * (1 - 0.05 * i),
                     fill=False, ec="#FFA500", alpha=0.08, lw=1, zorder=1))

    ax.add_patch(plt.Circle((mx, my), moon_r, color="#111111", zorder=2))
    ax.add_patch(plt.Circle((mx, my), moon_r, fill=False, ec="#333333", lw=1.5, zorder=3))

    if obs_frac > 0.85:
        rng = np.random.RandomState(42)
        n_stars = int(30 * (obs_frac - 0.85) / 0.15)
        sx = rng.uniform(-2.4, 2.4, n_stars)
        sy = rng.uniform(-2.4, 2.4, n_stars)
        ax.scatter(sx, sy, s=1, color="white", alpha=0.6, zorder=0)

    ax.set_title(
        f"Eclipse View:  {eclipse['date_raw']}  —  {ecl_type}\n"
        f"Observer: {obs_lat:.1f}°N, {obs_lon:.1f}°E   |   "
        f"{dist_km:,.0f} km from center   |   "
        f"Estimated obscuration: {obs_frac:.1%}",
        color="white", fontsize=11, fontweight="bold", pad=14,
    )
    info = f"Magnitude: {mag}   |   Saros: {eclipse.get('saros', '?')}"
    if path_w > 0:
        info += f"   |   Path width: {path_w:.0f} km"
    ax.text(0, -2.35, info, ha="center", color="#aaaaaa", fontsize=9)
    ax.axis("off")
    fig.patch.set_facecolor("#0e0e0e")
    plt.tight_layout()
    return fig


# ============================================================
# CHATBOT SEARCH HELPERS
# ============================================================

# Search, summary, and date-parsing methods are now on catalog.
# e.g. catalog.find_by_date(), catalog.find_next_visible(),
#      catalog.summary(), catalog.parse_date()


# ============================================================
# LLM CONTEXT BUILDER & CHAT
# ============================================================

SYSTEM_PROMPT = """You are an expert solar eclipse advisor. You help people find
and plan for solar eclipses based on a NASA catalog of 224 eclipses from 2001-2100.

When the user asks about eclipses, you will receive ECLIPSE DATA pulled from the
database as context. This app uses an approximate geometric model, so present
location-specific claims as estimates unless confidence is high.

Your capabilities:
- Tell users the next visible eclipse(s) from their location
- Describe what a specific eclipse will look like from a given place
- Provide viewing advice (safety, best locations along the path, weather tips)
- Explain eclipse types (Total, Annular, Hybrid, Partial) and what they look like
- Suggest the best lat/lon coordinates for viewing a given eclipse

IMPORTANT: When recommending a viewing location or responding about a specific
eclipse, ALWAYS include a line in this exact format so the app can update the
visualization automatically:
  VIEWER_SET: lat=XX.X, lon=XX.X, eclipse=YYYY Mon DD

For example:
  VIEWER_SET: lat=30.3, lon=-97.7, eclipse=2024 Apr 08

Keep answers concise but informative. Use the eclipse data provided - do not invent
eclipse dates or magnitudes.
When referring to local outcomes, prefer wording like "likely totality", "likely
partial", or "near path edge (lower confidence)" instead of absolute certainty."""

CITY_COORDS = {
    "new york": (40.7, -74.0), "los angeles": (34.1, -118.2),
    "chicago": (41.9, -87.6), "houston": (29.8, -95.4),
    "austin": (30.3, -97.7), "dallas": (32.8, -96.8),
    "denver": (39.7, -105.0), "seattle": (47.6, -122.3),
    "miami": (25.8, -80.2), "atlanta": (33.7, -84.4),
    "london": (51.5, -0.1), "paris": (48.9, 2.3),
    "tokyo": (35.7, 139.7), "sydney": (-33.9, 151.2),
    "cairo": (30.0, 31.2), "mumbai": (19.1, 72.9),
    "beijing": (39.9, 116.4), "mexico city": (19.4, -99.1),
    "toronto": (43.7, -79.4), "berlin": (52.5, 13.4),
    "rome": (41.9, 12.5), "madrid": (40.4, -3.7),
    "san francisco": (37.8, -122.4), "phoenix": (33.4, -112.0),
    "boston": (42.4, -71.1), "washington": (38.9, -77.0),
    "nashville": (36.2, -86.8), "portland": (45.5, -122.7),
    "indianapolis": (39.8, -86.2), "cleveland": (41.5, -81.7),
    "columbus": (39.96, -83.0), "cincinnati": (39.1, -84.5),
    # Frequently requested eclipse location (2017 path of totality)
    "madras": (44.63, -121.13), "madras oregon": (44.63, -121.13),
    "madras, oregon": (44.63, -121.13), "madras or": (44.63, -121.13),
}


def build_eclipse_context(user_message):
    msg = user_message.lower()
    context_parts = []
    obs_lat, obs_lon = None, None

    coord_patterns = [
        r'(\-?\d+\.?\d*)\s*[°]?\s*[NnSs]?\s*,?\s*(\-?\d+\.?\d*)\s*[°]?\s*[EeWw]?',
        r'lat(?:itude)?\s*[:=]?\s*(\-?\d+\.?\d*)\s*,?\s*lon(?:gitude)?\s*[:=]?\s*(\-?\d+\.?\d*)',
    ]
    for pat in coord_patterns:
        m = re.search(pat, user_message)
        if m:
            try:
                obs_lat = float(m.group(1))
                obs_lon = float(m.group(2))
                # Validate coordinate ranges
                if -90 <= obs_lat <= 90 and -180 <= obs_lon <= 180:
                    break
                else:
                    obs_lat, obs_lon = None, None
            except (ValueError, IndexError):
                obs_lat, obs_lon = None, None

    for city, (clat, clon) in CITY_COORDS.items():
        if city in msg:
            obs_lat, obs_lon = clat, clon
            context_parts.append(f"[Detected city: {city.title()} -> {clat} N, {clon} E]")
            break

    if any(kw in msg for kw in ["next", "upcoming", "when", "soonest", "future"]):
        if obs_lat is not None:
            results = catalog.find_next_visible(obs_lat, obs_lon, n=3)
            if results:
                context_parts.append(f"NEXT ECLIPSES VISIBLE FROM ({obs_lat} N, {obs_lon} E):")
                for ecl, dt, dist in results:
                    context_parts.append(catalog.summary(ecl, obs_lat, obs_lon))
                    context_parts.append("---")
            else:
                context_parts.append(f"No upcoming eclipses found visible from ({obs_lat}, {obs_lon}).")

    date_patterns = [
        r'(\d{4}\s+\w{3}\s+\d{1,2})',
        r'(\w+\s+\d{1,2},?\s+\d{4})',
        r'(\d{4})',
    ]
    for pat in date_patterns:
        m = re.search(pat, user_message)
        if m:
            date_str = m.group(1)
            matches = catalog.find_by_date(date_str)
            if matches:
                context_parts.append(f"ECLIPSES MATCHING '{date_str}':")
                for ecl in matches[:5]:
                    context_parts.append(catalog.summary(ecl, obs_lat, obs_lon))
                    context_parts.append("---")
            break

    for etype in ["total", "annular", "hybrid", "partial"]:
        if etype in msg:
            type_eclipses = [e for e in catalog.eclipses if e["type"].lower() == etype]
            context_parts.append(f"DATABASE: {len(type_eclipses)} {etype} eclipses in catalog.")
            now = datetime.now()
            upcoming = [(e, catalog.parse_date(e["date_raw"]))
                        for e in type_eclipses
                        if catalog.parse_date(e["date_raw"]) and
                           catalog.parse_date(e["date_raw"]) > now][:3]
            for e, dt in upcoming:
                context_parts.append(catalog.summary(e, obs_lat, obs_lon))
                context_parts.append("---")
            break

    if not context_parts:
        context_parts.append(
            f"DATABASE: {len(catalog)} solar eclipses from 2001-2100. "
            f"Types: Total, Annular, Hybrid, Partial. "
            f"Ask about a specific date, location, or eclipse type for detailed info."
        )

    return "\n".join(context_parts), obs_lat, obs_lon


def parse_viewer_set(reply):
    """
    Parse VIEWER_SET: lat=XX.X, lon=XX.X, eclipse=YYYY Mon DD from the LLM reply.
    Returns (lat, lon, eclipse_date_str) or (None, None, None).
    """
    m = re.search(
        r'VIEWER_SET:\s*lat\s*=\s*(\-?\d+\.?\d*)\s*[,;\s]\s*lon\s*=\s*(\-?\d+\.?\d*)\s*[,;\s]\s*eclipse\s*=\s*([^\n\r]+)',
        reply,
        flags=re.IGNORECASE,
    )
    if m:
        return float(m.group(1)), float(m.group(2)), m.group(3).strip()
    return None, None, None


# find_eclipse_index_by_date is now catalog.find_index_by_date()


def chat_with_llm(user_message, chat_history):
    context, obs_lat, obs_lon = build_eclipse_context(user_message)
    augmented_msg = f"{user_message}\n\n[ECLIPSE DATABASE CONTEXT]\n{context}"

    chat_history.append({"role": "user", "content": augmented_msg})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history

    if not api_key:
        return "Error: API key not configured. Please set ASTRO1221_API_KEY in your .env file.", obs_lat, obs_lon

    try:
        response = completion(
            model="openai/GPT-4.1-mini",
            messages=messages,
            api_base=API_BASE,
            api_key=api_key,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Sorry, I encountered an error: {e}"

    chat_history.append({"role": "assistant", "content": reply})
    return reply, obs_lat, obs_lon


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "viewer_lat" not in st.session_state:
    st.session_state.viewer_lat = 0.0
if "viewer_lon" not in st.session_state:
    st.session_state.viewer_lon = 0.0
if "viewer_eclipse_idx" not in st.session_state:
    st.session_state.viewer_eclipse_idx = 0
if "viewer_time" not in st.session_state:
    st.session_state.viewer_time = -1.0

# ============================================================
# UI LAYOUT
# ============================================================
st.title("🌒 Eclipse Explorer")
st.caption("Chat with the Eclipse Bot and explore estimated eclipse views from any location.")
st.info("Location-specific eclipse classifications are geometric estimates. Near path edges, uncertainty is higher.")

# Custom CSS: fixed-height scrollable chat container
st.markdown("""
<style>
    /* Make chat container scrollable with fixed height */
    div[data-testid="stVerticalBlock"] div[data-testid="stChatMessageContainer"] {
        max-height: 500px;
        overflow-y: auto;
    }
    /* Fixed-height container for the chat column */
    .chat-container {
        height: 700px;
        overflow-y: auto;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

col_chat, col_viz = st.columns([1, 1], gap="large")

# ---- LEFT COLUMN: CHATBOT ----
with col_chat:
    st.subheader("Eclipse Chatbot")
    st.markdown(
        "*Try: 'When is the next eclipse from Austin?' or "
        "'Tell me about the 2026 total eclipse'*"
    )

    # Scrollable chat container
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                display_text = re.sub(r'VIEWER_SET:.*', '', msg["content"]).strip()
                st.markdown(display_text)

    # Chat input (stays pinned below the container)
    if prompt := st.chat_input("Ask about an eclipse..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            reply, obs_lat, obs_lon = chat_with_llm(prompt, st.session_state.chat_history)

        # Parse VIEWER_SET from reply to update visualization
        v_lat, v_lon, v_eclipse = parse_viewer_set(reply)
        if v_lat is not None and v_lon is not None:
            # Validate coordinates before setting
            if -90 <= v_lat <= 90 and -180 <= v_lon <= 180:
                st.session_state.viewer_lat = v_lat
                st.session_state.viewer_lon = v_lon
                idx = catalog.find_index_by_date(v_eclipse)
                if idx is not None and 0 <= idx < len(catalog):
                    st.session_state.viewer_eclipse_idx = idx
                    st.session_state.viewer_time = 0.0
                    st.toast(f"Viewer updated: {v_lat:.1f}°N, {v_lon:.1f}°E", icon="📍")
                else:
                    st.toast(f"Coordinates set, but eclipse '{v_eclipse}' not found", icon="⚠️")
            else:
                st.toast("Invalid coordinates from chatbot", icon="⚠️")

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ---- RIGHT COLUMN: VISUALIZATION ----
with col_viz:
    st.subheader("Eclipse Viewer")

    # Eclipse dropdown
    eclipse_labels = catalog.labels
    # Ensure index is valid
    valid_idx = max(0, min(st.session_state.viewer_eclipse_idx, len(catalog) - 1))
    if valid_idx != st.session_state.viewer_eclipse_idx:
        st.session_state.viewer_eclipse_idx = valid_idx
    
    selected_idx = st.selectbox(
        "Eclipse:",
        range(len(catalog)),
        index=st.session_state.viewer_eclipse_idx,
        format_func=lambda i: eclipse_labels[i],
        key="eclipse_select",
    )
    
    # Validate selected_idx is within bounds
    if not (0 <= selected_idx < len(catalog)):
        selected_idx = 0

    # Sliders
    time_frac = st.slider(
        "Time (Moon position):",
        min_value=-1.0, max_value=1.0, step=0.02,
        value=st.session_state.viewer_time,
        key="time_slider",
    )
    lat = st.slider(
        "Latitude (°N):",
        min_value=-90.0, max_value=90.0, step=0.5,
        value=st.session_state.viewer_lat,
        key="lat_slider",
    )
    lon = st.slider(
        "Longitude (°E):",
        min_value=-180.0, max_value=180.0, step=0.5,
        value=st.session_state.viewer_lon,
        key="lon_slider",
    )

    # Draw the eclipse
    fig = draw_eclipse_sky(catalog[selected_idx], lat, lon, time_frac)
    st.pyplot(fig)
    plt.close(fig)

    # Show eclipse info
    ecl = catalog[selected_idx]
    raw = ecl.get("_raw", {})
    ecl_lat = parse_coord(raw.get("latitude", "0N"))
    ecl_lon = parse_coord(raw.get("longitude", "0E"))
    dist = haversine_km(lat, lon, ecl_lat, ecl_lon)

    st.markdown(
        f"**Eclipse center:** {ecl_lat:.1f}°N, {ecl_lon:.1f}°E  \n"
        f"**Your distance:** {dist:,.0f} km from center  \n"
        f"**Saros:** {ecl.get('saros', '?')}  |  "
        f"**Duration:** {ecl.get('duration', 'N/A')}"
    )

