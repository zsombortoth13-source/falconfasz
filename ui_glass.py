"""The see-through look for turtle_ui.py: a soft gradient backdrop with translucent, blurred surfaces (sidebar, cards, metrics,
expanders, inputs, buttons).

Streamlit's own theme (.streamlit/config.toml) has no transparency, so this is the one place custom CSS is used. Everything else
-- fonts, colours, radius, sizes -- stays in config.toml. Set ENABLED = False and the app falls back to the plain theme; nothing
else depends on this file.

Hooks used: Streamlit's data-testid attributes, and the .st-key-<key> classes of the three bordered containers in turtle_ui.py
(card_assets, card_settings, card_setup). A Streamlit update that renames a test id only costs the effect on that element.
"""

import streamlit as st

ENABLED = True

_LIGHT = {
    "backdrop": ("radial-gradient(900px 520px at 6% -8%, rgba(96,139,255,.30), transparent 62%),"
                 "radial-gradient(760px 520px at 96% 2%, rgba(167,139,250,.24), transparent 58%),"
                 "radial-gradient(900px 640px at 55% 108%, rgba(56,189,248,.20), transparent 62%),"
                 "linear-gradient(180deg,#f6f8fd 0%,#eef2fa 100%)"),
    "glass": "rgba(255,255,255,.58)", "soft": "rgba(255,255,255,.36)", "strong": "rgba(255,255,255,.80)",
    "side": "rgba(255,255,255,.50)", "line": "rgba(255,255,255,.85)", "hair": "rgba(15,23,42,.08)",
    "shadow": "0 10px 32px rgba(30,41,90,.10)", "input": "rgba(255,255,255,.72)", "thumb": "rgba(15,23,42,.22)",
    "header": "rgba(246,248,253,.55)", "glow": "rgba(37,99,235,.30)", "solid": "#f9fbff",
}
_DARK = {
    "backdrop": ("radial-gradient(900px 520px at 6% -8%, rgba(63,115,242,.38), transparent 62%),"
                 "radial-gradient(760px 520px at 96% 2%, rgba(139,92,246,.30), transparent 58%),"
                 "radial-gradient(900px 640px at 55% 108%, rgba(14,165,233,.20), transparent 62%),"
                 "linear-gradient(180deg,#0a0f18 0%,#0d1322 100%)"),
    "glass": "rgba(255,255,255,.055)", "soft": "rgba(255,255,255,.03)", "strong": "rgba(255,255,255,.10)",
    "side": "rgba(8,12,20,.55)", "line": "rgba(255,255,255,.11)", "hair": "rgba(255,255,255,.07)",
    "shadow": "0 12px 36px rgba(0,0,0,.40)", "input": "rgba(255,255,255,.07)", "thumb": "rgba(255,255,255,.22)",
    "header": "rgba(10,15,24,.55)", "glow": "rgba(63,115,242,.40)", "solid": "#141b27",
}

_CSS = """
:root{--g-backdrop:__backdrop__;--g-glass:__glass__;--g-soft:__soft__;--g-strong:__strong__;--g-side:__side__;--g-line:__line__;
--g-hair:__hair__;--g-shadow:__shadow__;--g-input:__input__;--g-thumb:__thumb__;--g-header:__header__;--g-glow:__glow__;--g-solid:__solid__;}
@media (prefers-reduced-transparency: reduce){:root{--g-glass:var(--g-solid);--g-soft:var(--g-solid);--g-strong:var(--g-solid);--g-side:var(--g-solid);--g-input:var(--g-solid);--g-header:var(--g-solid);}}
@supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))){:root{--g-glass:var(--g-solid);--g-soft:var(--g-solid);--g-strong:var(--g-solid);--g-side:var(--g-solid);--g-input:var(--g-solid);--g-header:var(--g-solid);}}

.stApp{background:var(--g-backdrop) !important;background-attachment:fixed !important;}
[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:transparent !important;}
[data-testid="stHeader"]{background:var(--g-header) !important;backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);}
[data-testid="stMainBlockContainer"]{max-width:1560px;padding:3.4rem 2.5rem 4rem;}
h1,h2,h3{letter-spacing:-0.02em;}
h1{letter-spacing:-0.03em;}

section[data-testid="stSidebar"]{background:var(--g-side) !important;backdrop-filter:blur(24px) saturate(160%);-webkit-backdrop-filter:blur(24px) saturate(160%);border-right:1px solid var(--g-line) !important;}
section[data-testid="stSidebar"] > div,[data-testid="stSidebarContent"]{background:transparent !important;}

.st-key-card_assets,.st-key-card_settings,.st-key-card_setup,[data-testid="stMetric"],[data-testid="stExpander"] details{
background:var(--g-glass) !important;border:1px solid var(--g-line) !important;border-radius:16px !important;box-shadow:var(--g-shadow);
backdrop-filter:blur(16px) saturate(150%);-webkit-backdrop-filter:blur(16px) saturate(150%);}
[class*="st-key-card_"] [data-testid="stExpander"] details,[class*="st-key-card_"] [data-testid="stMetric"]{
background:var(--g-soft) !important;box-shadow:none;backdrop-filter:none;-webkit-backdrop-filter:none;}
[data-testid="stMetric"]{padding:0;container-type:inline-size;}
[data-testid="stMetricValue"] p{font-size:clamp(1rem,12cqw,2rem) !important;white-space:nowrap !important;overflow:visible !important;text-overflow:clip !important;}
[data-testid="stMetricLabel"] p,[data-testid="stMetricDelta"] p{white-space:normal !important;overflow:visible !important;text-overflow:clip !important;overflow-wrap:anywhere;}

[data-testid="stTextInputRootElement"],[data-testid="stNumberInputContainer"],[data-testid="stSelectbox"] [data-baseweb="select"] > div,[data-testid="stMultiSelect"] [data-baseweb="select"] > div{
background:var(--g-input) !important;border-radius:10px !important;border:1px solid var(--g-hair) !important;}
[data-testid="stTextInputField"],[data-testid="stNumberInputField"]{background:transparent !important;}

[data-testid="stBaseButton-secondary"]{background:var(--g-glass) !important;border:1px solid var(--g-line) !important;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);transition:background .15s,transform .15s;}
[data-testid="stBaseButton-secondary"]:hover{background:var(--g-strong) !important;transform:translateY(-1px);}
[data-testid="stBaseButton-primary"]{box-shadow:0 6px 18px var(--g-glow);transition:transform .15s,box-shadow .15s;}
[data-testid="stBaseButton-primary"]:hover{transform:translateY(-1px);box-shadow:0 8px 22px var(--g-glow);}

[data-testid="stAlert"]{border-radius:14px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);}
[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid var(--g-hair);}
[data-testid="stTabs"] [role="tablist"]{border-bottom:1px solid var(--g-hair) !important;}

*::-webkit-scrollbar{width:8px;height:8px;}
*::-webkit-scrollbar-thumb{background:var(--g-thumb);border-radius:8px;}
*::-webkit-scrollbar-track{background:transparent;}
"""


def is_dark() -> bool:
    """Whether the viewer's Streamlit theme is dark (False when it cannot be told, e.g. in tests)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:   # an older Streamlit, or a context without a theme
        return False


def inject():
    """Write the stylesheet for the viewer's current light/dark theme; does nothing when ENABLED is False."""
    if not ENABLED:
        return
    css = _CSS
    for name, value in (_DARK if is_dark() else _LIGHT).items():
        css = css.replace(f"__{name}__", value)
    st.html(f"<style>{css}</style>")
