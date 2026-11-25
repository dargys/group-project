"""Checkout assistant: suggests which fruit or vegetable is on the scale.

Pilot with nine kinds of fruit and vegetables.

Run from the project root:
    streamlit run app.py
"""
import io
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.predict import CLASS_NAMES, load_cnn_model, predict_image


# Below this confidence (in %), the customer is asked to check the suggestions carefully
CONFIDENCE_LIMIT = 60

DISPLAY_NAMES = {name: name.replace("_", " ").capitalize() for name in CLASS_NAMES}
MANUAL_SEARCH = "manual search"


st.set_page_config(page_title="Checkout Assistant", page_icon="🛒", layout="wide")


@st.cache_resource
def get_model():
    # Load the trained CNN once and reuse it for every photo
    return load_cnn_model()


def label(name):
    return f"{DISPLAY_NAMES[name]}"


# --- Demo settings (not part of the customer's screen) -------------------------------
with st.sidebar:
    st.header("Demo settings")
    source = st.radio("Photo source", ["Upload a photo", "Use the camera"])
    st.caption(
        "In a store, the photo would come from a camera mounted above the scale. "
        "In this demo it is uploaded or taken with a phone or laptop camera."
    )
    st.divider()
    st.caption("Pilot items: " + ", ".join(DISPLAY_NAMES.values()) + ".")

# --- Customer screen -----------------------------------------------------------------
st.title("🛒 Checkout Assistant")
st.write("Place your fruit or vegetable on the scale and choose the right item below.")

if source == "Upload a photo":
    photo_file = st.file_uploader("Photo from the scale", type=["jpg", "jpeg", "png"])
else:
    photo_file = st.camera_input("Photo from the scale")

if photo_file is None:
    st.info("Waiting for an item on the scale…")
    st.stop()

# Start over when a new photo arrives
photo_id = getattr(photo_file, "file_id", None) or f"{photo_file.name}-{photo_file.size}"
if st.session_state.get("photo_id") != photo_id:
    st.session_state["photo_id"] = photo_id
    st.session_state["registered"] = None

@st.cache_data(show_spinner="Identifying the item…")
def classify(photo_bytes):
    # Phone photos are often stored sideways with an orientation tag: turn them upright
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(photo_bytes))).convert("RGB")
    # Phone photos are large, and the model only needs 64 × 48 pixels
    image.thumbnail((1024, 1024))
    prediction, probabilities = predict_image(image, get_model())
    return image, prediction, probabilities


photo, prediction, probabilities = classify(photo_file.getvalue())
ranking = pd.Series(probabilities).sort_values(ascending=False)
top_three = ranking.head(3)

photo_column, choice_column = st.columns([2, 3], gap="large")

with photo_column:
    st.image(photo, caption="Photo from the scale")

with choice_column:
    registered = st.session_state["registered"]

    if registered is None:
        st.subheader("Is it one of these?", anchor=False)
        if top_three.iloc[0] < CONFIDENCE_LIMIT:
            st.warning("We are not sure. Please check the suggestions carefully.")

        for rank, (name, score) in enumerate(top_three.items(), start=1):
            with st.container(border=True):
                text_column, button_column = st.columns([3, 1], vertical_alignment="center")
                with text_column:
                    st.subheader(label(name), anchor=False)
                    st.caption(f"Suggestion {rank} · {score:.0f}% likely")
                with button_column:
                    if st.button("Select", key=f"select_{name}"):
                        st.session_state["registered"] = name
                        st.rerun()

        if st.button("None of these – search manually"):
            st.session_state["registered"] = MANUAL_SEARCH
            st.rerun()

    elif registered == MANUAL_SEARCH:
        st.subheader("Search manually", anchor=False)
        st.info("The customer searches the menu as today.")
        if st.button("Back to the suggestions"):
            st.session_state["registered"] = None
            st.rerun()

    else:
        st.success("Item registered")
        st.header(label(registered), anchor=False)
        if st.button("Undo"):
            st.session_state["registered"] = None
            st.rerun()

# --- Explanation for the presentation --------------------------------------------------
with st.expander("How the suggestion was made"):
    st.write(
        "A convolutional neural network (CNN) trained on 1,603 photos gives each of the "
        "nine pilot items a probability. The app shows the three most likely items, so the "
        "right one is usually among them even when the first suggestion is wrong."
    )
    st.bar_chart(ranking.rename(index={name: DISPLAY_NAMES[name] for name in ranking.index}))
    st.caption(
        "The model can only choose between the nine pilot items. Photos from other cameras, "
        "backgrounds or lighting can give lower accuracy."
    )
