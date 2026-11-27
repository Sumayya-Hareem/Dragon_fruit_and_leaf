"""
Streamlit demo: upload (or photograph) a dragon fruit or leaf and see what each trained model predicts.

Run from the repo root:
    pip install -r app/requirements.txt
    streamlit run app/app.py

Checkpoints are not stored in git. Copy them from the training outputs into app/models/
(see app/models/README.md), or point DRAGON_MODELS_DIR at another folder.
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import ARMS, CLASSES, available_arms, load_arm, predict_arm

MODELS_DIR = Path(os.environ.get("DRAGON_MODELS_DIR", Path(__file__).resolve().parent / "models"))
LOW_CONFIDENCE = 0.60

st.set_page_config(page_title="Dragon fruit & leaf classifier", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_arm(key: str, models_dir: str):
    return load_arm(key, Path(models_dir))


def show_result(key: str, image: Image.Image):
    try:
        arm = get_arm(key, str(MODELS_DIR))
    except Exception as e:
        st.error(str(e))
        return
    probs = predict_arm(arm, image)              # (n_checkpoints, 4)
    mean = probs.mean(0)
    top = int(mean.argmax())
    n = len(arm.names)

    st.markdown(f"**Arm {key}**: {arm.spec.title}")
    st.markdown(f"### {CLASSES[top]}")
    detail = f"{mean[top]:.0%} confidence"
    if n > 1:
        detail += f" (average of {n} seeds; {int((probs.argmax(1) == top).sum())} of {n} agree)"
    st.caption(detail)
    if mean[top] < LOW_CONFIDENCE:
        st.warning("Low confidence: treat this prediction with caution.")
    for c, p in zip(CLASSES, mean):
        st.progress(float(p), text=f"{c}: {p:.0%}")
    if n > 1:
        with st.expander("Per-seed predictions"):
            table = pd.DataFrame(probs, columns=CLASSES, index=arm.names)
            table.insert(0, "prediction", [CLASSES[i] for i in probs.argmax(1)])
            st.dataframe(table.round(3))


def main():
    st.title("Dragon fruit & leaf health classifier")
    st.caption("Part of a deep learning portfolio series. A research demo, not agricultural advice.")

    found = available_arms(MODELS_DIR)
    if not found:
        st.info(
            f"No model checkpoints found in `{MODELS_DIR}`.\n\n"
            "Copy the `best.pt` files from the training runs into that folder, one sub-folder per run, "
            "for example `tl_pre_224_s0/best.pt` (arm B). See `app/models/README.md`."
        )
        st.stop()

    default = ["B"] if "B" in found else [next(iter(found))]
    chosen = st.sidebar.multiselect(
        "Models to run", options=list(found), default=default,
        format_func=lambda k: f"{k}: {ARMS[k].title}",
        help="Select several to compare models on the same photo.")
    st.sidebar.caption("Test macro-F1 (mean ± SD over 5 seeds, 677 test images)")
    for k in chosen:
        st.sidebar.write(f"**{k}**: {ARMS[k].test_f1} ({len(found[k])} checkpoint(s) found)")

    tab_upload, tab_camera = st.tabs(["Upload a photo", "Use the camera"])
    with tab_upload:
        uploaded = st.file_uploader("Dragon fruit or leaf photo", type=["jpg", "jpeg", "png", "webp"])
    with tab_camera:
        shot = st.camera_input("Take a photo")
    source = uploaded or shot

    if source is None:
        st.info("Upload or take a photo to get a prediction.")
    elif not chosen:
        st.info("Choose at least one model in the sidebar.")
    else:
        try:
            image = Image.open(source)
            image.load()
        except (UnidentifiedImageError, OSError):
            st.error("That file could not be read as an image.")
            st.stop()
        left, right = st.columns([1, 2])
        with left:
            st.image(image, caption="Your photo")
        with right:
            cols = st.columns(len(chosen))
            for col, key in zip(cols, chosen):
                with col:
                    show_result(key, image)

    with st.expander("About these models and their limits"):
        st.markdown(
            """
- **Four classes only:** Bad fruit, Bad leaf, Good fruit, Good leaf. The models have no "not a dragon fruit"
  option, so any photo, including an unrelated one, is assigned one of the four.
- **Training data:** one public dataset of dragon fruit and leaf images from Bangladesh, trained without
  data augmentation. Photos from other cameras, backgrounds or varieties may behave differently.
- **Typical confusion:** the largest error cells are Good leaf versus Bad leaf. Fruit classes had the
  fewest training images (198 each).
- **Preprocessing** mirrors the training images: the photo is shrunk and re-encoded as JPEG before it is
  resized to the network's input size.
- **Scores in the sidebar** are test macro-F1 on the full 677-image test set, which includes near-duplicates
  of training images (see the README, "Data quality").
            """
        )


main()
