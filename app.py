"""
app.py
------
OTT-Style Movie Recommendation Web Application
Built with Streamlit + Hybrid Recommendation Engine (SVD-CF + Genre-CB)

Run:  streamlit run app.py
"""

import os
import sys
import base64
from pathlib import Path
from urllib.parse import quote_plus

# Make sure src/ and visualization/ are importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.data_preprocessing import preprocess_pipeline
from src.hybrid_recommender import HybridRecommender
from src.poster_service import get_poster_url, get_poster_url_by_title, clear_cache
from visualization.plots import (
    plot_rating_distribution,
    plot_genre_distribution,
    plot_top_movies,
    plot_top_rated_movies,
    plot_user_activity,
    plot_ratings_over_time,
    plot_hybrid_weight_comparison,
)

# ============================================================
# Page config  (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="CineAI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — 100% Netflix faithful
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Netflix+Sans:wght@300;400;500;700&display=swap');

    /* ═══════════════════════════════════════════════════════
       GLOBAL RESET — pure Netflix black
    ═══════════════════════════════════════════════════════ */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section.main > div {
        background-color: #141414 !important;
        color: #e5e5e5;
        font-family: 'Netflix Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Hide Streamlit default chrome (Deploy/menu/header badge) */
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    #MainMenu,
    header {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Kill Streamlit chrome padding */
    [data-testid="stMain"] > div:first-child { padding-top: 0 !important; }
    .block-container { padding: 0 4% 40px 4% !important; max-width: 100% !important; }

    /* ═══════════════════════════════════════════════════════
       SIDEBAR — Netflix dark panel
    ═══════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #2a2a2a !important;
    }
    [data-testid="stSidebar"] * { color: #e5e5e5 !important; }
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background: #1a1a1a !important;
        color: #fff !important;
        border: 1px solid #404040 !important;
        border-radius: 4px !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #1a1a1a !important;
        border: 1px solid #404040 !important;
    }

    /* ── Sidebar nav buttons ── */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: #e5e5e5 !important;
        border: none !important;
        border-radius: 4px !important;
        text-align: left !important;
        padding: 10px 14px !important;
        width: 100% !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        letter-spacing: 0.3px !important;
        margin-bottom: 2px !important;
        transition: background 0.12s !important;
        transform: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ═══════════════════════════════════════════════════════
       GLOBAL BUTTONS  (main content area)
    ═══════════════════════════════════════════════════════ */
    .stButton > button {
        background: #e50914 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 9px 20px !important;
        cursor: pointer !important;
        transition: background 0.12s, transform 0.08s !important;
    }
    .stButton > button:hover {
        background: #f40612 !important;
        transform: scale(1.03) !important;
    }

    /* ── Play / More-Info button pair ── */
    .nf-btn-play {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ffffff;
        color: #000 !important;
        border: none;
        border-radius: 4px;
        font-size: 15px;
        font-weight: 700;
        padding: 10px 24px;
        cursor: pointer;
        margin-right: 10px;
        transition: background 0.12s, opacity 0.12s;
        text-decoration: none;
    }
    .nf-btn-play:hover { background: rgba(255,255,255,0.75); }
    .nf-btn-info {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(109,109,110,0.7);
        color: #fff !important;
        border: none;
        border-radius: 4px;
        font-size: 15px;
        font-weight: 700;
        padding: 10px 24px;
        cursor: pointer;
        transition: background 0.12s;
        text-decoration: none;
    }
    .nf-btn-info:hover { background: rgba(109,109,110,0.4); }

    /* ═══════════════════════════════════════════════════════
       HERO BANNER — full-bleed, no border-radius
    ═══════════════════════════════════════════════════════ */
    .nf-hero {
        position: relative;
        width: calc(100% + 8%);
        margin-left: -4%;
        margin-right: -4%;
        min-height: 56vw;
        max-height: 700px;
        background-size: cover;
        background-position: center top;
        overflow: hidden;
        margin-bottom: 0;
    }
    /* vignette: left fill + bottom fade into #141414 */
    .nf-hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(to right,  rgba(20,20,20,0.98) 0%,  rgba(20,20,20,0.68) 38%, transparent 75%),
            linear-gradient(to top,    rgba(20,20,20,1.00) 0%,  rgba(20,20,20,0.00) 28%);
    }
    /* gradient colour border separating hero from rows */
    .nf-hero-fade {
        position: relative;
        margin-top: -120px;
        height: 120px;
        background: linear-gradient(to bottom, transparent, #141414);
        z-index: 2;
        margin-left: -4%;
        margin-right: -4%;
        width: calc(100% + 8%);
    }
    .nf-hero-body {
        position: absolute;
        bottom: 28%;
        left: 4%;
        z-index: 3;
        max-width: 46%;
    }
    .nf-hero-label {
        font-size: 11px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #e50914;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .nf-hero-title {
        font-size: clamp(2rem, 4vw, 3.4rem);
        font-weight: 700;
        line-height: 1.05;
        color: #fff;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        margin-bottom: 14px;
    }
    .nf-hero-desc {
        font-size: 1rem;
        color: #d2d2d2;
        line-height: 1.6;
        margin-bottom: 20px;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.9);
    }
    .nf-hero-meta {
        font-size: 13px;
        color: #bcbcbc;
        margin-bottom: 22px;
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        align-items: center;
    }
    .nf-match { color: #46d369; font-weight: 700; font-size: 14px; }
    .nf-age   {
        border: 1px solid #bcbcbc;
        padding: 1px 6px;
        font-size: 12px;
        color: #bcbcbc;
    }

    /* ═══════════════════════════════════════════════════════
       ROW LABELS — exact Netflix typography
    ═══════════════════════════════════════════════════════ */
    .nf-row-label {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e5e5e5;
        margin: 28px 0 8px 0;
        letter-spacing: 0;
    }
    .nf-row-label:hover { color: #bababa; }

    /* ═══════════════════════════════════════════════════════
       MOVIE CARD — Netflix thumbnail style
    ═══════════════════════════════════════════════════════ */
    .nf-card {
        position: relative;
        border-radius: 4px;
        overflow: hidden;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease, z-index 0s 0.2s;
        z-index: 1;
        background: #181818;
    }
    .nf-card:hover {
        transform: scale(1.06);
        box-shadow: 0 0 24px 8px rgba(0,0,0,0.75);
        z-index: 10;
        border-radius: 6px 6px 0 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .nf-card:focus-within {
        outline: 2px solid #ffffff;
        outline-offset: 2px;
    }
    .nf-thumb {
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
        display: block;
        border-radius: 4px;
    }
    .nf-thumb-placeholder {
        width: 100%;
        aspect-ratio: 2/3;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 12px;
        border-radius: 4px;
        text-align: center;
        gap: 6px;
    }
    .nf-thumb-placeholder .pt { font-size: 12px; font-weight: 700; color: rgba(255,255,255,0.85); line-height: 1.3; word-break: break-word; }

    /* always-visible metadata strip for each card */
    .nf-card-meta {
        padding: 8px 8px 9px;
        background: #181818;
        border-top: 1px solid #262626;
    }
    .nf-card-meta .nf-card-title {
        margin-bottom: 3px;
    }
    .nf-card-meta .nf-meta-line {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        font-size: 10px;
        color: #bcbcbc;
    }
    .nf-card-meta .nf-meta-line .nf-rating {
        color: #f5c518;
        font-weight: 700;
    }

    /* hover info panel that slides up under the thumb */
    .nf-card-info {
        display: none;
        background: #181818;
        border-radius: 0 0 6px 6px;
        padding: 10px 10px 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.8);
        position: relative;
        z-index: 10;
    }
    .nf-card:hover .nf-card-info { display: block; }
    .nf-card-title { font-size: 12px; font-weight: 700; color: #fff; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .nf-card-row   { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-top: 4px; }
    .nf-green  { color: #46d369; font-size: 11px; font-weight: 700; }
    .nf-tag {
        border: 1px solid #808080;
        border-radius: 3px;
        font-size: 9px;
        padding: 1px 4px;
        color: #ddd;
    }
    .nf-genre { font-size: 10px; color: #bcbcbc; }

    /* ═══════════════════════════════════════════════════════
       HORIZONTAL SCROLL RAIL
    ═══════════════════════════════════════════════════════ */
    [data-testid="stMain"] [data-testid="stHorizontalBlock"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        gap: 4px !important;
        padding-bottom: 8px !important;
        scrollbar-width: none !important;
    }
    [data-testid="stMain"] [data-testid="stHorizontalBlock"]::-webkit-scrollbar { display: none !important; }
    [data-testid="stMain"] [data-testid="stHorizontalBlock"] > div {
        min-width: 140px !important;
        flex-shrink: 0 !important;
    }

    /* Rail navigation controls */
    .nf-rail-nav {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 8px;
        margin: 2px 0 10px;
    }
    .nf-rail-count {
        font-size: 11px;
        color: #9a9a9a;
        letter-spacing: 0.4px;
        margin-right: 8px;
    }
    .nf-rail-nav .stButton > button {
        background: rgba(0,0,0,0.55) !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 4px !important;
        min-width: 38px !important;
        height: 32px !important;
        padding: 0 !important;
        font-size: 14px !important;
        line-height: 1 !important;
        opacity: 0.16 !important;
        transition: opacity 0.18s, background 0.12s !important;
    }
    .nf-rail-nav:hover .stButton > button { opacity: 0.95 !important; }
    .nf-rail-nav .stButton > button:disabled { opacity: 0.08 !important; }
    .nf-rail-nav .stButton > button:hover {
        background: rgba(229,9,20,0.82) !important;
        border-color: rgba(229,9,20,1) !important;
        transform: none !important;
        opacity: 1 !important;
    }

    /* Keyboard accessibility polish */
    .stButton > button:focus-visible,
    [data-testid="stSidebar"] .stButton > button:focus-visible,
    .stTextInput input:focus-visible,
    .stSelectbox [role="combobox"]:focus-visible {
        outline: 2px solid #ffffff !important;
        outline-offset: 2px !important;
        box-shadow: none !important;
    }

    /* ═══════════════════════════════════════════════════════
       DETAIL / SCORE BOXES
    ═══════════════════════════════════════════════════════ */
    .detail-poster {
        width: 100%; height: 320px; border-radius: 4px;
        display: flex; align-items: center; justify-content: center;
        flex-direction: column;
    }
    .detail-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 6px; color: #fff; }
    .detail-meta  { color: #999; font-size: 0.9rem; margin-bottom: 12px; }
    .score-box {
        background: #181818;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 14px 18px;
        text-align: center;
    }
    .score-val { font-size: 1.8rem; font-weight: 700; color: #e50914; }
    .score-lbl { font-size: 11px; color: #808080; margin-top: 2px; }

    /* ── Movie detail cinematic header ── */
    .detail-hero {
        position: relative;
        min-height: 320px;
        overflow: hidden;
        margin-bottom: 28px;
        background-size: cover;
        background-position: center 25%;
        width: calc(100% + 8%);
        margin-left: -4%;
        margin-right: -4%;
    }
    .detail-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(to right, rgba(20,20,20,0.96) 0%, rgba(20,20,20,0.7) 45%, rgba(20,20,20,0.25) 100%),
            linear-gradient(to top,   rgba(20,20,20,1.00) 0%, transparent 35%);
    }
    .detail-hero-body {
        position: relative;
        z-index: 1;
        padding: 52px 4%;
        max-width: 60%;
    }
    .detail-hero-body h2 { font-size: 2.6rem; margin: 0 0 10px 0; line-height: 1.06; color: #fff; }
    .detail-hero-body .detail-meta { color: #c0c0c0; margin-bottom: 14px; }
    .detail-hero-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
    .detail-chip {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 3px;
        padding: 4px 10px;
        font-size: 12px;
        color: #e5e5e5;
    }
    .detail-chip.red {
        background: rgba(229,9,20,0.15);
        border-color: rgba(229,9,20,0.4);
        color: #ff8080;
    }

    /* ═══════════════════════════════════════════════════════
       MISC
    ═══════════════════════════════════════════════════════ */
    .metric-card {
        background: #181818;
        border-radius: 4px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2a2a2a;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #e50914; }
    .metric-lbl { font-size: 12px; color: #808080; margin-top: 4px; }

    .search-result-row {
        background: #181818;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border: 1px solid #2a2a2a;
        cursor: pointer;
        transition: border-color 0.12s;
    }
    .search-result-row:hover { border-color: #e50914; }

    .badge {
        display: inline-block;
        background: #2a2a2a;
        color: #c0c0c0;
        font-size: 9px;
        padding: 2px 6px;
        border-radius: 3px;
        margin-right: 3px;
        margin-top: 3px;
    }
    h1,h2,h3,h4 { color: #fff !important; }
    hr { border-color: #2a2a2a; }
    .stSlider > div { color: #e5e5e5; }
    .stSelectbox > div > div { background: #181818 !important; color: #e5e5e5 !important; }
    .stTextInput > div > div > input {
        background: #181818 !important;
        color: #fff !important;
        border: 1px solid #404040 !important;
        border-radius: 4px !important;
    }

    /* section header / caption (kept for My Recs and Analytics) */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e5e5e5;
        margin: 28px 0 6px 0;
    }
    .section-caption { margin: 0 0 10px 0; color: #808080; font-size: 12px; }

    /* hero-tag / hero-chip (kept for detail page) */
    .hero-tag {
        display: inline-block;
        font-size: 10px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #e50914;
        font-weight: 700;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Session state initialisation
# ============================================================
_DEFAULTS = {
    "page": "Home",
    "selected_movie_id": None,
    "search_query": "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Auto-load TMDB key from environment so users do not need to paste it every run.
if "tmdb_api_key" not in st.session_state:
    st.session_state.tmdb_api_key = (
        os.getenv("TMDB_API_KEY", "")
        or os.getenv("TMDB_V3_API_KEY", "")
    ).strip()


# ============================================================
# Load & cache the recommendation engine
# ============================================================
@st.cache_resource(show_spinner="Loading recommendation engine …")
def load_recommender() -> HybridRecommender:
    data = preprocess_pipeline()
    rec  = HybridRecommender(n_factors=50)
    rec.fit(data)
    return rec




rec = load_recommender()
data = rec.data

# movieId -> tmdbId (from links.csv)
_TMDB_MAP: dict = data.get("tmdb_map", {})


# ============================================================
# Utility: poster gradient & card HTML
# ============================================================
_GRADIENTS = [
    ("1a237e", "6a1b9a"),
    ("1b5e20", "006064"),
    ("b71c1c", "880e4f"),
    ("e65100", "37474f"),
    ("0d47a1", "1a237e"),
    ("880e4f", "4a148c"),
    ("006064", "0d47a1"),
    ("33691e", "1b5e20"),
    ("37474f", "263238"),
    ("4a148c", "880e4f"),
    ("bf360c", "e65100"),
    ("006064", "004d40"),
]

def _gradient(movie_id: int) -> str:
    c1, c2 = _GRADIENTS[movie_id % len(_GRADIENTS)]
    return f"linear-gradient(135deg, #{c1}, #{c2})"


def _short(text: str, n: int = 22) -> str:
    return text[:n] + "…" if len(text) > n else text


def _genre_badges(genres: str) -> str:
    parts = genres.split("|")[:3]
    return "".join(f'<span class="badge">{g}</span>' for g in parts)


def _resolve_poster_url(movie_id: int, title: str) -> str | None:
    """Resolve poster using tmdbId first, then title search fallback."""
    tmdb_id = _TMDB_MAP.get(movie_id)
    tmdb_key = st.session_state.get("tmdb_api_key", "").strip()
    if not tmdb_key:
        return None

    poster_url = get_poster_url(tmdb_id, tmdb_key) if tmdb_id else None
    if not poster_url:
        poster_url = get_poster_url_by_title(title, tmdb_key)
    return poster_url


def _trailer_search_url(title: str) -> str:
    """Return YouTube trailer search URL for a movie title."""
    q = quote_plus(f"{title} official trailer")
    return f"https://www.youtube.com/results?search_query={q}"


def _inject_hotkeys() -> None:
    """Enable lightweight keyboard shortcuts: J=prev rail, K=next rail."""
    components.html(
        """
        <script>
        (() => {
            const doc = window.parent.document;
            if (window.parent.__cineaiHotkeysBound) return;
            window.parent.__cineaiHotkeysBound = true;

            const clickArrow = (symbol) => {
                const buttons = Array.from(doc.querySelectorAll('button'))
                    .filter((b) => b.textContent && b.textContent.trim() === symbol && !b.disabled);
                if (buttons.length > 0) {
                    buttons[0].click();
                }
            };

            doc.addEventListener('keydown', (e) => {
                const t = e.target;
                const tag = t && t.tagName ? t.tagName.toLowerCase() : '';
                if (tag === 'input' || tag === 'textarea' || (t && t.isContentEditable)) return;

                if (e.key === 'j' || e.key === 'J') {
                    e.preventDefault();
                    clickArrow('❮');
                }
                if (e.key === 'k' || e.key === 'K') {
                    e.preventDefault();
                    clickArrow('❯');
                }
            });
        })();
        </script>
        """,
        height=0,
    )


def render_movie_card(movie_id: int, title: str, genres: str,
                      avg_rating: float, num_ratings: int, key: str) -> None:
    """Render Netflix-style thumbnail card with hover info panel."""
    poster_url = _resolve_poster_url(movie_id, title)
    first_genre = genres.split("|")[0] if genres else "Movie"
    match_pct   = min(99, max(60, int(avg_rating / 5.0 * 99)))
    short_title = _short(title, 26)

    if poster_url:
        thumb_html = (
            f'<img class="nf-thumb" src="{poster_url}" alt="{title}" '
            f'onerror="this.style.display=\'none\';">'
        )
    else:
        grad = _gradient(movie_id)
        thumb_html = (
            f'<div class="nf-thumb-placeholder" style="background:{grad};">'
            f'<div class="pt">{_short(title, 28)}</div>'
            f'</div>'
        )

    card_html = (
        f'<div class="nf-card">'
        f'{thumb_html}'
        f'<div class="nf-card-meta">'
        f'<div class="nf-card-title">{short_title}</div>'
        f'<div class="nf-meta-line">'
        f'<span>{first_genre}</span>'
        f'<span class="nf-rating">⭐ {avg_rating:.1f}</span>'
        f'</div>'
        f'</div>'
        f'<div class="nf-card-info">'
        f'<div class="nf-card-title">{short_title}</div>'
        f'<div class="nf-card-row">'
        f'<span class="nf-green">{match_pct}% Match</span>'
        f'<span class="nf-tag">{"HD"}</span>'
        f'</div>'
        f'<div class="nf-card-row"><span class="nf-genre">{first_genre}</span></div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    trailer_url = _trailer_search_url(title)
    st.link_button("▶ Trailer", trailer_url, use_container_width=True)
    if st.button("ℹ More Info", key=f"{key}_info", help=title, use_container_width=True):
        st.session_state.selected_movie_id = movie_id
        st.session_state.page = "Movie Details"
        st.rerun()


def render_movie_grid(movies_df: pd.DataFrame, cols: int = 8, key_prefix: str = "card") -> None:
    """Render a horizontal rail of movie cards with prev/next controls."""
    if movies_df.empty:
        st.info("No movies to display.")
        return

    total = len(movies_df)
    start_key = f"{key_prefix}_start"
    if start_key not in st.session_state:
        st.session_state[start_key] = 0

    max_start = max(0, total - cols)
    cur_start = int(st.session_state[start_key])
    cur_start = min(max(0, cur_start), max_start)
    st.session_state[start_key] = cur_start

    st.markdown('<div class="nf-rail-nav">', unsafe_allow_html=True)
    n1, n2, n3 = st.columns([1.0, 1.0, 8.0])
    with n3:
        end_idx = min(cur_start + cols, total)
        st.markdown(
            f'<div class="nf-rail-count">{cur_start + 1}-{end_idx} of {total}</div>',
            unsafe_allow_html=True,
        )
    with n1:
        if st.button("❮", key=f"{key_prefix}_prev", disabled=cur_start == 0):
            st.session_state[start_key] = max(0, cur_start - cols)
            st.rerun()
    with n2:
        if st.button("❯", key=f"{key_prefix}_next", disabled=cur_start >= max_start):
            st.session_state[start_key] = min(max_start, cur_start + cols)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    row = movies_df.iloc[cur_start:cur_start + cols]
    columns = st.columns(max(len(row), 1))
    for col, (_, m) in zip(columns, row.iterrows()):
        with col:
            render_movie_card(
                int(m["movieId"]), m["title"], m["genres"],
                float(m.get("avg_rating", 0)),
                int(m.get("num_ratings", 0)),
                key=f"{key_prefix}_{int(m['movieId'])}",
            )


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    # Netflix-style wordmark
    st.markdown(
        '<div style="font-size:2rem;font-weight:900;letter-spacing:-1px;color:#e50914;'
        'margin-bottom:4px;font-family:Arial Black,Arial,sans-serif;">CINEAI</div>'
        '<div style="font-size:11px;color:#808080;margin-bottom:16px;letter-spacing:1.5px;text-transform:uppercase;">Recommendation Engine</div>',
        unsafe_allow_html=True,
    )

    # Profile avatar row
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">'
        '<div style="width:32px;height:32px;border-radius:4px;background:#e50914;'
        'display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;">U</div>'
        '<span style="font-size:13px;color:#e5e5e5;">My Profile</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:#2a2a2a;margin:8px 0 12px 0;">', unsafe_allow_html=True)

    _NAV_ITEMS = [
        ("🏠", "Home",               "Home"),
        ("🔍", "Search",             "Search"),
        ("🎥", "Movie Details",      "Movie Details"),
        ("🌟", "My Recommendations", "My Recommendations"),
        ("📊", "Analytics",          "Analytics"),
    ]
    for _icon, _label, _pkey in _NAV_ITEMS:
        if st.session_state.page == _pkey:
            st.markdown(
                f'<div style="background:rgba(229,9,20,0.15);border-left:3px solid #e50914;'
                f'border-radius:0 4px 4px 0;padding:10px 14px;color:#fff;font-weight:700;'
                f'font-size:14px;margin-bottom:2px;">'
                f'{_icon}&nbsp;&nbsp;{_label}</div>',
                unsafe_allow_html=True,
            )
        else:
            if st.button(f"{_icon}  {_label}", key=f"nav_{_pkey}", use_container_width=True):
                st.session_state.page = _pkey
                st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    valid_users = sorted(rec.valid_user_ids)
    default_uid = valid_users[0] if valid_users else 1
    user_id = st.selectbox(
        "Select User ID",
        valid_users,
        index=0,
        help="Choose a user to generate personalised recommendations.",
    )

    alpha = st.slider(
        "Hybrid Weight (α)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="α=1 → pure Collaborative Filtering | α=0 → pure Content-Based",
    )

    n_recs = st.slider(
        "Number of Recommendations",
        min_value=5, max_value=30, value=10, step=5,
    )

    st.markdown("---")
    st.markdown("### 📋 Dataset Info")
    st.metric("Movies",   f"{data['n_movies']:,}")
    st.metric("Ratings",  f"{data['n_ratings']:,}")
    st.metric("Users",    f"{data['n_users']:,}")

    st.markdown("---")
    st.markdown("### 🖼️ Movie Posters")
    tmdb_key_input = st.text_input(
        "TMDB API Key",
        value=st.session_state.get("tmdb_api_key", ""),
        type="password",
        placeholder="Paste free TMDB API key here …",
        help=(
            "1. Sign up free at themoviedb.org\n"
            "2. Go to Settings → API\n"
            "3. Copy the v3 API Key and paste here.\n\n"
            "Without a key, coloured gradient placeholders are shown."
        ),
    )
    if tmdb_key_input != st.session_state.get("tmdb_api_key", ""):
        st.session_state.tmdb_api_key = tmdb_key_input
        clear_cache()   # reset poster cache when key changes
        st.rerun()

    if st.session_state.get("tmdb_api_key", "").strip():
        st.success("✅ Real posters enabled")
    else:
        st.info("ℹ️ Add TMDB key to load real movie posters")

    st.markdown("---")
    st.caption("Hotkeys: J = previous rail, K = next rail")
    st.caption("Powered by SVD-CF + Genre-CB Hybrid")

# Enable global rail hotkeys (J/K).
_inject_hotkeys()


# ============================================================
# PAGE: Home
# ============================================================
if st.session_state.page == "Home":
    recs_df = rec.recommend_for_user(user_id, n=n_recs, alpha=alpha)
    trending_df = rec.get_trending(n=10)

    # Featured hero movie (recommendation first, trending fallback)
    if not recs_df.empty:
        featured = recs_df.iloc[0]
    else:
        featured = trending_df.iloc[0]

    featured_mid = int(featured["movieId"])
    featured_title = str(featured["title"])
    featured_genres = str(featured.get("genres", ""))
    featured_rating = float(featured.get("avg_rating", 0.0))
    featured_count = int(featured.get("num_ratings", 0))
    featured_poster = _resolve_poster_url(featured_mid, featured_title)
    featured_bg     = featured_poster if featured_poster else _gradient(featured_mid)

    _featured_match = min(99, max(60, int(featured_rating / 5.0 * 99)))
    _featured_genre = featured_genres.split('|')[0] if featured_genres else 'Movie'
    _bg_style       = f"url('{featured_poster}')" if featured_poster else featured_bg

    st.markdown(
        f'<div class="nf-hero" style="background-image:{_bg_style};">'
        f'<div class="nf-hero-body">'
        f'<div class="nf-hero-label">CineAI Pick for You</div>'
        f'<div class="nf-hero-title">{featured_title}</div>'
        f'<div class="nf-hero-meta">'
        f'<span class="nf-match">{_featured_match}% Match</span>'
        f'<span class="nf-age">PG-13</span>'
        f'<span>{_featured_genre}</span>'
        f'<span>⭐ {featured_rating:.1f}&nbsp;·&nbsp;{featured_count:,} ratings</span>'
        f'</div>'
        f'<div class="nf-hero-desc">Personalised by CineAI hybrid engine — blending collaborative signals from similar viewers with genre-DNA content analysis.</div>'
        f'<div class="nf-hero-btns" id="hero-btn-row"></div>'
        f'</div></div>'
        f'<div class="nf-hero-fade"></div>',
        unsafe_allow_html=True,
    )

    # Hero action row: native Streamlit controls anchored over the hero.
    st.markdown(
        '<style>'
        '.hero-action-row { display:flex; gap:10px; margin-top:-80px; margin-bottom:40px;'
        ' padding-left:4%; position:relative; z-index:10; }'
        '.hero-action-row .stButton > button {'
        '  border-radius:4px !important; font-size:15px !important; font-weight:700 !important;'
        '  padding:10px 26px !important; letter-spacing:0 !important;'
        '  transition: opacity 0.12s !important; }'
        '.hero-action-row [data-testid="stLinkButton"] a {'
        '  border-radius:4px !important; font-size:15px !important; font-weight:700 !important;'
        '  padding:10px 26px !important; letter-spacing:0 !important; text-decoration:none !important;'
        '  border:none !important; box-shadow:none !important; min-height:auto !important; }'
        '.hero-action-row .stButton:first-child > button {'
        '  background:#ffffff !important; color:#000000 !important; }'
        '.hero-action-row .stButton:first-child > button:hover {'
        '  background:rgba(255,255,255,0.75) !important; transform:none !important; }'
        '.hero-action-row [data-testid="stLinkButton"] a {'
        '  background:#ffffff !important; color:#000000 !important; }'
        '.hero-action-row [data-testid="stLinkButton"] a:hover {'
        '  background:rgba(255,255,255,0.75) !important; color:#000000 !important; }'
        '.hero-action-row .stButton:last-child > button {'
        '  background:rgba(109,109,110,0.7) !important; color:#ffffff !important; }'
        '.hero-action-row .stButton:last-child > button:hover {'
        '  background:rgba(109,109,110,0.4) !important; transform:none !important; }'
        '</style>'
        '<div class="hero-action-row">',
        unsafe_allow_html=True,
    )
    trailer_url = _trailer_search_url(featured_title)
    _hcol1, _hcol2, _hpad = st.columns([1.2, 1.6, 8])
    with _hcol1:
        st.link_button("▶  Play", trailer_url, use_container_width=False)
    with _hcol2:
        if st.button("ℹ  More Info", key="hero_info"):
            st.session_state.selected_movie_id = featured_mid
            st.session_state.page = "Movie Details"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Top Picks ────────────────────────────────────────────────
    st.markdown('<div class="nf-row-label">🌟 Top Picks For You</div>', unsafe_allow_html=True)
    if recs_df.empty:
        st.info("Not enough data for personalised recommendations. Try a different user.")
    else:
        render_movie_grid(recs_df.head(10), cols=10, key_prefix="home_rec")

    # ── Trending ──────────────────────────────────────────────────
    st.markdown('<div class="nf-row-label">🔥 Trending This Week</div>', unsafe_allow_html=True)
    render_movie_grid(trending_df, cols=10, key_prefix="home_trend")

    # ── Popular ───────────────────────────────────────────────────
    st.markdown('<div class="nf-row-label">👥 Most Watched</div>', unsafe_allow_html=True)
    popular_df = rec.get_popular(n=10)
    render_movie_grid(popular_df, cols=10, key_prefix="home_pop")

    # ── Top Rated ─────────────────────────────────────────────────
    st.markdown('<div class="nf-row-label">⭐ Critics’ Picks</div>', unsafe_allow_html=True)
    top_rated_df = rec.get_top_rated(min_ratings=3, n=10)
    render_movie_grid(top_rated_df, cols=10, key_prefix="home_top")

    # ── Similar Movies ─────────────────────────────────────────────
    st.markdown('<div class="nf-row-label">🎯 Similar Movies</div>', unsafe_allow_html=True)
    home_sim_df = rec.recommend_similar_to_movie(featured_mid, n=10, alpha=0.3)
    render_movie_grid(home_sim_df, cols=10, key_prefix="home_sim")


# ============================================================
# PAGE: Search
# ============================================================
elif st.session_state.page == "Search":
    st.markdown("# 🔍 Search Movies")
    st.markdown("Find any movie and explore recommendations based on it.")

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        query = st.text_input(
            "Movie title",
            value=st.session_state.search_query,
            placeholder="e.g. Inception, Dark Knight, Toy Story …",
            label_visibility="collapsed",
        )
    with col_btn:
        do_search = st.button("🔍 Search", use_container_width=True)

    if query:
        st.session_state.search_query = query
        results = rec.search_movies(query)

        if results.empty:
            st.warning(f"No movies found matching **{query}**. Try a different keyword.")
        else:
            st.markdown(f'<div style="color:#bcbcbc;font-size:13px;margin-bottom:8px;">{len(results)} result(s) found</div>', unsafe_allow_html=True)
            render_movie_grid(results.head(20), cols=10, key_prefix="search")

            # Show recommendations based on the top matched movie.
            mid = int(results.iloc[0]["movieId"])
            st.markdown('<div class="nf-row-label">🎯 Recommended From Your Search</div>', unsafe_allow_html=True)
            sim = rec.recommend_similar_to_movie(mid, n=10, alpha=0.3)
            render_movie_grid(sim, cols=10, key_prefix="search_sim")
    else:
        st.markdown('<div class="nf-row-label">📈 Popular on CineAI</div>', unsafe_allow_html=True)
        all_movies = data["movie_stats"].sort_values("popularity_score", ascending=False)
        render_movie_grid(all_movies.head(20), cols=10, key_prefix="browse")


# ============================================================
# PAGE: Movie Details
# ============================================================
elif st.session_state.page == "Movie Details":
    mid = st.session_state.selected_movie_id

    if mid is None:
        st.info("Select a movie from Home or Search to view its details.")
    else:
        detail = rec.get_movie_detail(mid)

        if detail is None:
            st.error("Movie not found.")
        else:
            # Back button
            if st.button("← Back"):
                st.session_state.page = "Home"
                st.rerun()

            # ── Cinematic hero banner ──────────────────────────────
            d_mid     = int(detail["movieId"])
            _dg_str   = str(detail.get("genres", ""))
            _dp_bg    = _resolve_poster_url(d_mid, str(detail["title"]))
            _dh_bg    = f"url('{_dp_bg}')" if _dp_bg else _gradient(d_mid)
            _dh_chips = "".join(
                f'<span class="detail-chip">{g}</span>'
                for g in _dg_str.split("|")[:4] if g
            )
            st.markdown(
                f'<div class="detail-hero" style="background-image:{_dh_bg};'
                f'background-color:{"#0f0f0f" if _dp_bg else ""};">'
                f'<div class="detail-hero-body">'
                f'<div class="hero-tag">Now Viewing</div>'
                f'<h2>{detail["title"]}</h2>'
                f'<div class="detail-meta">{_dg_str.replace("|", " · ") or "Movie"}</div>'
                f'<div class="detail-hero-chips">'
                f'<span class="detail-chip red">⭐ {detail.get("avg_rating","N/A")} / 5</span>'
                f'<span class="detail-chip">👥 {int(detail.get("num_ratings", 0)):,} ratings</span>'
                f'{_dh_chips}'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

            # Layout: poster | info
            c1, c2 = st.columns([1, 2])

            with c1:
                poster_d = _resolve_poster_url(d_mid, str(detail["title"]))
                if poster_d:
                    st.markdown(
                        f'<div style="border-radius:10px;overflow:hidden;border:1px solid #2a2a2a;">'
                        f'<img src="{poster_d}" alt="{detail["title"]}" '
                        f'style="width:100%;border-radius:10px;display:block;" '
                        f'onerror="this.style.display=\'none\';">'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    grad = _gradient(d_mid)
                    st.markdown(
                        f'<div class="detail-poster" style="background:{grad};">'
                        f'<div style="font-size:16px;font-weight:700;opacity:0.9;">Poster unavailable</div>'
                        f'<div style="color:rgba(255,255,255,0.85);font-size:14px;'
                        f'font-weight:700;margin-top:10px;text-align:center;padding:0 12px;">'
                        f'{detail["title"]}</div></div>',
                        unsafe_allow_html=True,
                    )

            with c2:
                st.markdown(f'<div class="detail-title">{detail["title"]}</div>', unsafe_allow_html=True)
                genres_str = detail.get("genres", "")
                st.markdown(
                    f'<div class="detail-meta">{_genre_badges(genres_str)}</div>',
                    unsafe_allow_html=True,
                )

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(
                        f'<div class="score-box"><div class="score-val">{detail.get("avg_rating","N/A")}</div>'
                        f'<div class="score-lbl">Avg Rating</div></div>',
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f'<div class="score-box"><div class="score-val">{int(detail.get("num_ratings",0))}</div>'
                        f'<div class="score-lbl">Total Ratings</div></div>',
                        unsafe_allow_html=True,
                    )
                with m3:
                    pop_pct = detail.get("popularity_score", 0)
                    st.markdown(
                        f'<div class="score-box"><div class="score-val">{pop_pct:.0%}</div>'
                        f'<div class="score-lbl">Popularity</div></div>',
                        unsafe_allow_html=True,
                    )
                with m4:
                    pred_rating = rec._cf.predict_rating(user_id, mid)
                    st.markdown(
                        f'<div class="score-box"><div class="score-val">'
                        f'{"N/A" if pred_rating == 0 else f"{pred_rating:.1f}"}</div>'
                        f'<div class="score-lbl">Your Predicted</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Genre list
                st.markdown("**Genres:**")
                genre_parts = genres_str.split("|") if genres_str else []
                st.markdown("  ·  ".join(f"`{g}`" for g in genre_parts))

                # Predict & recommend CTA
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🌟 Get Personalised Picks Based on This Movie"):
                    st.session_state.page = "My Recommendations"
                    st.rerun()

            # ── Similar Movies ────────────────────────────────────────
            st.markdown('<div class="nf-row-label">🎯 Similar Movies</div>', unsafe_allow_html=True)
            sim_df = rec.recommend_similar_to_movie(mid, n=10, alpha=0.3)
            if sim_df.empty:
                st.info("No similar movies found.")
            else:
                render_movie_grid(sim_df, cols=10, key_prefix=f"detail_sim_{mid}")

            # ── Rating distribution for this movie ────────────────────
            movie_ratings = data["ratings"][data["ratings"]["movieId"] == mid]["rating"]
            if len(movie_ratings) > 0:
                st.markdown('<div class="nf-row-label">📊 Rating Distribution</div>', unsafe_allow_html=True)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor("#141414")
                ax.set_facecolor("#181818")
                bins = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25]
                ax.hist(movie_ratings, bins=bins, color="#e50914", edgecolor="#141414")
                ax.tick_params(colors="#fff")
                ax.set_xlabel("Rating", color="#fff")
                ax.set_ylabel("Count", color="#fff")
                ax.set_title(f"Ratings for: {detail['title'][:40]}", color="#fff", fontsize=11)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)


# ============================================================
# PAGE: My Recommendations
# ============================================================
elif st.session_state.page == "My Recommendations":
    st.markdown(f"# 🌟 Personalised Recommendations")
    st.markdown(f"*Showing top **{n_recs}** picks for **User {user_id}** (α = {alpha:.2f})*")

    # Alpha explainer
    with st.expander("ℹ️ How Hybrid Scoring Works", expanded=False):
        st.markdown(
            f"""
            **Hybrid Formula:**
            > `Final Score = {alpha:.2f} × Collaborative Score + {1-alpha:.2f} × Content Score`

            | Weight | Effect |
            |--------|--------|
            | **α = 1.0** | Pure Collaborative Filtering — recommends based on similar users' preferences |
            | **α = 0.5** | Balanced hybrid — blends both signals equally |
            | **α = 0.0** | Pure Content-Based — recommends based on genre similarity to your liked movies |

            Adjust **α** using the sidebar slider to tune the balance.
            """
        )

    recs_df = rec.recommend_for_user(user_id, n=n_recs, alpha=alpha)

    if recs_df.empty:
        st.warning("Not enough rating history for this user. Try a user ID with more ratings.")
    else:
        # Summary scores table
        with st.expander("📋 View Scores Table", expanded=False):
            display_df = recs_df[["title", "genres", "avg_rating", "cf_score", "cb_score", "hybrid_score"]].copy()
            display_df.columns = ["Title", "Genres", "Avg Rating", "CF Score", "CB Score", "Hybrid Score"]
            display_df = display_df.round(4)
            st.dataframe(display_df, use_container_width=True)

        st.markdown('<div class="nf-row-label">🎬 Your Top Picks</div>', unsafe_allow_html=True)
        render_movie_grid(recs_df, cols=10, key_prefix=f"myrec_{user_id}")

        # ── Hybrid weight comparison chart ────────────────────────────
        st.markdown('<div class="nf-row-label">⚖️ How Alpha Affects Your Recommendations</div>',
                    unsafe_allow_html=True)
        with st.spinner("Generating comparison chart …"):
            try:
                fig = plot_hybrid_weight_comparison(rec, user_id, n=10)
                st.pyplot(fig, use_container_width=True)
            except Exception:
                st.info("Not enough data to render the comparison chart for this user.")

    # ── What the user has already rated ───────────────────────────
    st.markdown('<div class="nf-row-label">📼 Your Watch History</div>', unsafe_allow_html=True)
    user_history = (
        data["ratings"][data["ratings"]["userId"] == user_id]
        .sort_values("rating", ascending=False)
        .head(10)
        .merge(data["movie_stats"][["movieId", "title", "genres", "avg_rating", "num_ratings"]], on="movieId", how="left")
    )
    if user_history.empty:
        st.info("No rating history found for this user.")
    else:
        render_movie_grid(user_history, cols=10, key_prefix=f"hist_{user_id}")


# ============================================================
# PAGE: Analytics
# ============================================================
elif st.session_state.page == "Analytics":
    st.markdown("# 📊 Dataset Analytics")
    st.markdown("Explore insights about the MovieLens dataset powering CineAI.")

    # Top metrics row
    st.markdown("### 🔢 Overview")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{data["n_movies"]:,}</div>'
            f'<div class="metric-lbl">Total Movies</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{data["n_ratings"]:,}</div>'
            f'<div class="metric-lbl">Total Ratings</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{data["n_users"]:,}</div>'
            f'<div class="metric-lbl">Unique Users</div></div>', unsafe_allow_html=True)
    with m4:
        avg_r = data["ratings"]["rating"].mean()
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{avg_r:.2f}★</div>'
            f'<div class="metric-lbl">Mean Rating</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Rating distribution | Genre distribution
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Rating Distribution")
        fig1 = plot_rating_distribution(data["ratings"])
        st.pyplot(fig1, use_container_width=True)
    with col_b:
        st.markdown("#### Genre Distribution")
        fig2 = plot_genre_distribution(data["movies"])
        st.pyplot(fig2, use_container_width=True)

    st.markdown("---")

    # Row 2: Top movies | Top rated
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### Most Rated Movies")
        fig3 = plot_top_movies(data["movie_stats"], n=15)
        st.pyplot(fig3, use_container_width=True)
    with col_d:
        st.markdown("#### Highest Rated Movies")
        fig4 = plot_top_rated_movies(data["movie_stats"], min_ratings=3, n=15)
        st.pyplot(fig4, use_container_width=True)

    st.markdown("---")

    # Row 3: User activity | Ratings over time
    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("#### Most Active Users")
        fig5 = plot_user_activity(data["ratings"], top_n=20)
        st.pyplot(fig5, use_container_width=True)
    with col_f:
        st.markdown("#### Ratings Over Time")
        fig6 = plot_ratings_over_time(data["ratings"])
        st.pyplot(fig6, use_container_width=True)

    st.markdown("---")

    # Evaluation metrics (computed on demand)
    st.markdown("### 🎯 Model Evaluation")
    with st.expander("Run Evaluation (may take a moment)", expanded=False):
        if st.button("▶ Run Evaluation Now"):
            with st.spinner("Evaluating CF model …"):
                from src.evaluation import evaluate_cf, evaluate_ranking
                cf_metrics = evaluate_cf(rec._cf, data["ratings"])
                rank_metrics = evaluate_ranking(rec, data["ratings"], k=10)

            st.markdown("**Hold-out Rating Prediction (80/20 split):**")
            e1, e2, e3 = st.columns(3)
            with e1:
                st.metric("RMSE", f"{cf_metrics['rmse']:.4f}" if cf_metrics['rmse'] else "N/A")
            with e2:
                st.metric("MAE",  f"{cf_metrics['mae']:.4f}"  if cf_metrics['mae']  else "N/A")
            with e3:
                st.metric("Test Samples", f"{cf_metrics['n_test_samples']:,}")

            st.markdown("**Top-10 Ranking Quality (sampled users):**")
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Precision@10", f"{rank_metrics['avg_precision_at_k']:.4f}"
                          if rank_metrics['avg_precision_at_k'] else "N/A")
            with r2:
                st.metric("Recall@10", f"{rank_metrics['avg_recall_at_k']:.4f}"
                          if rank_metrics['avg_recall_at_k'] else "N/A")
            with r3:
                st.metric("F1@10", f"{rank_metrics['avg_f1_at_k']:.4f}"
                          if rank_metrics['avg_f1_at_k'] else "N/A")
            with r4:
                st.metric("Users Evaluated", rank_metrics["users_evaluated"])

