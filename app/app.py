"""
Streamlit demo: upload (or photograph) a dragon fruit or leaf and get a prediction.

Run from the repo root:
    pip install -r app/requirements.txt
    streamlit run app/app.py

Checkpoints are not stored in git. Copy them from the training outputs into app/models/
(see app/models/README.md), or point DRAGON_MODELS_DIR at another folder.

Background photo: put an image named background.jpg (or .jpeg / .png / .webp) in app/assets/.
Without one, a dragon-fruit coloured gradient is used.

Developer view: open the app with ?dev=1 to choose models, compare arms and see per-seed output.
The normal view always uses arm B (all available seeds, probabilities averaged).
"""
import base64
import io
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from inference import ARMS, CLASSES, available_arms, load_arm, predict_arm, to_rgb

HERE = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("DRAGON_MODELS_DIR", HERE / "models"))
ASSETS_DIR = HERE / "assets"
DEFAULT_ARM = "B"
LOW_CONFIDENCE = 0.60
TITLE = "Dragon Fruit & Leaf Health Classifier"


# ----------------------------------------------------------------------------- artwork

def _seed_points(n: int, scale: float, cx: float, cy: float):
    """Seeds laid out on a sunflower spiral, like the black seeds in a cut dragon fruit."""
    for i in range(1, n + 1):
        r = scale * math.sqrt(i)
        a = i * 2.39996323
        yield cx + r * math.cos(a), cy + r * math.sin(a), math.degrees(a)


def logo_svg() -> str:
    parts = ['<svg viewBox="-14 -14 128 128" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">']
    for k in range(12):                                    # green scales around the skin
        ang = k * 30 + 15
        x = 50 + 37 * math.cos(math.radians(ang))
        y = 50 + 37 * math.sin(math.radians(ang))
        parts.append(
            f'<path d="M0 0 C-6 -7 -5 -15 0 -23 C5 -15 6 -7 0 0Z" fill="#43b868" '
            f'transform="translate({x:.1f} {y:.1f}) rotate({ang + 90})"/>')
    parts.append('<circle cx="50" cy="50" r="38" fill="#d81b78"/>')
    parts.append('<circle cx="50" cy="50" r="31" fill="#fff3f8"/>')
    for x, y, a in _seed_points(52, 3.6, 50, 50):
        parts.append(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="1.6" ry="1.05" fill="#1b1220" '
                     f'transform="rotate({a:.0f} {x:.1f} {y:.1f})"/>')
    parts.append("</svg>")
    return "".join(parts)


def page_icon() -> Image.Image:
    size = 128
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 124, 124), fill=(216, 27, 120, 255))
    d.ellipse((20, 20, 108, 108), fill=(255, 243, 248, 255))
    for x, y, _ in _seed_points(52, 7.0, 64, 64):
        d.ellipse((x - 3, y - 2, x + 3, y + 2), fill=(27, 18, 32, 255))
    return img


@st.cache_resource(show_spinner=False)
def background_image_css() -> str:
    """CSS `background-image` value: dark overlay + photo from app/assets, or a gradient fallback."""
    overlay = ("linear-gradient(180deg, rgba(20,6,17,.58) 0%, rgba(20,6,17,.50) 40%, rgba(20,6,17,.82) 100%)")
    for name in ("background.jpg", "background.jpeg", "background.png", "background.webp"):
        path = ASSETS_DIR / name
        if not path.exists():
            continue
        try:
            im = to_rgb(ImageOps.exif_transpose(Image.open(path)))
            im.thumbnail((1920, 1920))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"{overlay}, url(data:image/jpeg;base64,{b64})"
        except Exception:
            break
    return ("radial-gradient(900px 620px at 12% 6%, rgba(200,30,115,.55), transparent 62%), "
            "radial-gradient(820px 620px at 92% 94%, rgba(36,150,76,.42), transparent 62%), "
            "linear-gradient(135deg, #250a1e 0%, #14081b 100%)")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background-image: __BG__;
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    color: #fff;
}
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
__HIDE_SIDEBAR__
.block-container { max-width: 940px; padding-top: 2.4rem; padding-bottom: 3.5rem; }

/* hero */
.dfc-hero { text-align: center; margin: 1.2rem 0 2rem; }
.dfc-logo svg { width: 92px; height: 92px; filter: drop-shadow(0 8px 22px rgba(0,0,0,.45)); }
.dfc-title { font-weight: 800; font-size: clamp(2rem, 5.2vw, 3.3rem); line-height: 1.08; letter-spacing: -.02em;
             margin: .9rem 0 .7rem; color: #fff; text-shadow: 0 4px 28px rgba(0,0,0,.5); }
.dfc-title span { background: linear-gradient(90deg, #ff7ab8 0%, #ff3d9a 55%, #ffa6d2 100%);
                  -webkit-background-clip: text; background-clip: text; color: transparent; text-shadow: none; }
.dfc-sub { max-width: 560px; margin: 0 auto; font-size: 1.08rem; line-height: 1.55; color: rgba(255,255,255,.88);
           text-shadow: 0 2px 14px rgba(0,0,0,.5); }

/* glass cards */
.st-key-uploadcard, .dfc-card {
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.20);
    border-radius: 22px;
    -webkit-backdrop-filter: blur(18px); backdrop-filter: blur(18px);
    box-shadow: 0 24px 64px rgba(0,0,0,.38);
}
.st-key-uploadcard { padding: 1.3rem 1.5rem 1.6rem; }
.dfc-card { padding: 1.5rem 1.6rem; }

/* tabs (selectors for both the newer react-aria markup and the older baseweb markup) */
[role="tablist"] { gap: .5rem !important; border-bottom: none !important; box-shadow: none !important;
    background: transparent !important; padding-bottom: .5rem; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], [data-testid="stTab"] > div[data-rac],
[role="tablist"]::before, [role="tablist"]::after { display: none !important; }
[data-testid="stTabs"] > div:first-child, [role="tablist"] + div[data-rac] { border-bottom: none !important; }
[data-testid="stTab"], [data-baseweb="tab"] { height: auto !important; padding: .5rem 1.25rem !important;
    border-radius: 999px !important; background: rgba(255,255,255,.10) !important;
    color: rgba(255,255,255,.88) !important; }
[data-testid="stTab"] p, [data-baseweb="tab"] p { font-weight: 600; font-size: .95rem; color: inherit !important; margin: 0; }
[data-testid="stTab"][aria-selected="true"], [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #d81b78, #ff4fa3) !important; color: #fff !important; }

/* uploader + camera */
[data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.07); border: 2px dashed rgba(255,255,255,.42);
    border-radius: 18px; padding: 2rem 1.2rem; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #ff6fb0; background: rgba(255,255,255,.12); }
[data-testid="stFileUploaderDropzone"] * { color: rgba(255,255,255,.92) !important; }
[data-testid="stFileUploaderDropzone"] small { color: rgba(255,255,255,.62) !important; }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"], [data-testid="stCameraInput"] button {
    background: linear-gradient(90deg, #d81b78, #ff4fa3); color: #fff !important; border: 0; border-radius: 999px;
    padding: .5rem 1.3rem; font-weight: 600; }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover { filter: brightness(1.1); }
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-minimal"],
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-borderlessIcon"] {
    background: rgba(255,255,255,.14); border: 0; border-radius: 999px; color: #fff !important; }
[data-testid="stFileChip"] { background: rgba(255,255,255,.14) !important; border-radius: 12px; }
[data-testid="stFileUploaderFile"] * { color: #fff !important; }
[data-testid="stCameraInput"] > div { border-radius: 16px; overflow: hidden; }
[data-testid="stCameraInput"] label p { color: #fff; }

/* result */
.dfc-photo { border-radius: 22px; overflow: hidden; border: 1px solid rgba(255,255,255,.22);
    box-shadow: 0 24px 64px rgba(0,0,0,.4); line-height: 0; background: rgba(0,0,0,.25); }
.dfc-photo img { width: 100%; max-height: 460px; object-fit: contain; }
.dfc-kicker { font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 700;
    color: rgba(255,255,255,.62); }
.dfc-label { font-size: clamp(1.9rem, 4.2vw, 2.6rem); font-weight: 800; letter-spacing: -.02em; margin: .25rem 0 .1rem; }
.dfc-card.good .dfc-label { color: #5ee58f; }
.dfc-card.bad .dfc-label { color: #ff6b7f; }
.dfc-conf { color: rgba(255,255,255,.85); font-size: 1rem; margin: .2rem 0 1rem; }
.dfc-conf b { color: #fff; font-size: 1.15rem; }
.dfc-note { margin: 0 0 1rem; padding: .7rem .9rem; border-radius: 12px; font-size: .92rem; line-height: 1.45;
    background: rgba(255,196,77,.16); border: 1px solid rgba(255,196,77,.45); color: #ffe3a3; }
.dfc-row { margin: .62rem 0; }
.dfc-row-top { display: flex; justify-content: space-between; font-size: .93rem; color: rgba(255,255,255,.88); margin-bottom: .28rem; }
.dfc-row.top .dfc-row-top { color: #fff; font-weight: 700; }
.dfc-bar { height: 9px; border-radius: 999px; background: rgba(255,255,255,.14); overflow: hidden; }
.dfc-bar span { display: block; height: 100%; border-radius: 999px; background: rgba(255,255,255,.55); }
.dfc-card.good .dfc-row.top .dfc-bar span { background: linear-gradient(90deg, #2fbf6b, #7df0a8); }
.dfc-card.bad  .dfc-row.top .dfc-bar span { background: linear-gradient(90deg, #e0405a, #ff8a98); }
</style>
"""

HIDE_SIDEBAR = ('[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], '
                '[data-testid="collapsedControl"] { display: none !important; }')


def inject_css(dev: bool):
    css = CSS.replace("__BG__", background_image_css()).replace("__HIDE_SIDEBAR__", "" if dev else HIDE_SIDEBAR)
    st.markdown(css, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- model

@st.cache_resource(show_spinner="Loading the model...")
def get_arm(key: str, models_dir: str):
    return load_arm(key, Path(models_dir))


def photo_data_uri(image: Image.Image, max_side: int = 900) -> str:
    im = to_rgb(ImageOps.exif_transpose(image))
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def verdict_html(mean) -> str:
    top = int(mean.argmax())
    tone = "good" if CLASSES[top].startswith("Good") else "bad"
    note = ""
    if mean[top] < LOW_CONFIDENCE:
        note = ('<div class="dfc-note">Not very sure about this one. A clear, well-lit photo of a single '
                'fruit or leaf works best.</div>')
    rows = []
    for i in sorted(range(len(CLASSES)), key=lambda j: -mean[j]):
        cls = "dfc-row top" if i == top else "dfc-row"
        rows.append(f'<div class="{cls}"><div class="dfc-row-top"><span>{CLASSES[i]}</span>'
                    f'<span>{mean[i]:.0%}</span></div>'
                    f'<div class="dfc-bar"><span style="width:{max(mean[i], 0.01) * 100:.1f}%"></span></div></div>')
    return (f'<div class="dfc-card {tone}"><div class="dfc-kicker">Result</div>'
            f'<div class="dfc-label">{CLASSES[top]}</div>'
            f'<div class="dfc-conf"><b>{mean[top]:.0%}</b> confident</div>'
            f'{note}{"".join(rows)}</div>')


def show_result_dev(key: str, image: Image.Image):
    """Original engineering view: one column per arm, with per-seed predictions."""
    try:
        arm = get_arm(key, str(MODELS_DIR))
    except Exception as e:
        st.error(str(e))
        return
    probs = predict_arm(arm, image)
    mean = probs.mean(0)
    top = int(mean.argmax())
    n = len(arm.names)
    st.markdown(f"**Arm {key}**: {arm.spec.title}")
    st.markdown(f"### {CLASSES[top]}")
    detail = f"{mean[top]:.0%} confidence"
    if n > 1:
        detail += f" (average of {n} seeds; {int((probs.argmax(1) == top).sum())} of {n} agree)"
    st.caption(detail)
    table = pd.DataFrame(probs, columns=CLASSES, index=arm.names)
    table.insert(0, "prediction", [CLASSES[i] for i in probs.argmax(1)])
    st.dataframe(table.round(3))


# ----------------------------------------------------------------------------- page

def main():
    st.set_page_config(page_title=TITLE, page_icon=page_icon(), layout="centered")
    dev = st.query_params.get("dev") == "1"
    inject_css(dev)

    st.markdown(
        f'<div class="dfc-hero"><div class="dfc-logo">{logo_svg()}</div>'
        f'<div class="dfc-title" role="heading" aria-level="1">Dragon Fruit &amp; Leaf<br>'
        f'<span>Health Classifier</span></div>'
        f'<div class="dfc-sub">Upload a photo of a dragon fruit or a leaf and the model tells you whether '
        f'it looks good or bad.</div></div>',
        unsafe_allow_html=True)

    found = available_arms(MODELS_DIR)
    if not found:
        st.info(
            f"No model checkpoints found in `{MODELS_DIR}`.\n\n"
            "Copy the `best.pt` files from the training runs into that folder, one sub-folder per run, "
            "for example `tl_pre_224_s0/best.pt`. See `app/models/README.md`."
        )
        st.stop()

    chosen = [DEFAULT_ARM if DEFAULT_ARM in found else next(iter(found))]
    if dev:
        chosen = st.sidebar.multiselect(
            "Models to run (developer view)", options=list(found), default=chosen,
            format_func=lambda k: f"{k}: {ARMS[k].title}") or chosen
        for k in chosen:
            st.sidebar.write(f"**{k}**: test macro-F1 {ARMS[k].test_f1} ({len(found[k])} checkpoint(s))")

    try:
        arm = get_arm(chosen[0], str(MODELS_DIR))      # load now so the first prediction is quick
    except Exception as e:
        st.error("The model could not be loaded." + (f"\n\n{e}" if dev else ""))
        st.stop()

    with st.container(key="uploadcard"):
        tab_upload, tab_camera = st.tabs(["Upload a photo", "Use the camera"])
        with tab_upload:
            uploaded = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"],
                                        label_visibility="collapsed")
        with tab_camera:
            shot = st.camera_input("Take a photo", label_visibility="collapsed")
    source = uploaded or shot
    if source is None:
        return

    try:
        image = Image.open(source)
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error("That file could not be read as an image. Please try a JPG or PNG photo.")
        return

    with st.spinner("Looking at your photo..."):
        probs = predict_arm(arm, image)
    mean = probs.mean(0)

    st.write("")
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown(f'<div class="dfc-photo"><img src="{photo_data_uri(image)}" alt="Your photo"></div>',
                    unsafe_allow_html=True)
    with right:
        st.markdown(verdict_html(mean), unsafe_allow_html=True)

    if dev:
        with st.expander("Developer view"):
            cols = st.columns(len(chosen))
            for col, key in zip(cols, chosen):
                with col:
                    show_result_dev(key, image)


main()
