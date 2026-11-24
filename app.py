"""Fruit and vegetable identifier.

Run from the project root:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.predict import CLASS_NAMES, load_cnn_model, predict_image


# Below this confidence (in %), the app tells the user that the model is unsure
CONFIDENCE_LIMIT = 60

DISPLAY_NAMES = {name: name.replace("_", " ").capitalize() for name in CLASS_NAMES}


st.set_page_config(page_title="Fruit and Vegetable Identifier", page_icon="🍎")


@st.cache_resource
def get_model():
    # Load the trained CNN once and reuse it for every photo
    return load_cnn_model()


st.title("Fruit and Vegetable Identifier")
st.write(
    "Upload a photo or take one with your camera. The model recognizes nine kinds of "
    "fruit and vegetables: " + ", ".join(DISPLAY_NAMES.values()) + "."
)

# --- Get a photo -------------------------------------------------------------
source = st.radio("Photo source", ["Upload a photo", "Use the camera"], horizontal=True)

if source == "Upload a photo":
    photo_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
else:
    photo_file = st.camera_input("Take a photo")

if photo_file is None:
    st.stop()

# Phone photos are often stored sideways with an orientation tag: turn them upright
photo = ImageOps.exif_transpose(Image.open(photo_file)).convert("RGB")

# --- Classify ----------------------------------------------------------------
prediction, probabilities = predict_image(photo, get_model())
ranking = pd.Series(probabilities).sort_values(ascending=False)
best_name, best_score = ranking.index[0], ranking.iloc[0]

photo_column, result_column = st.columns(2)

with photo_column:
    st.image(photo, caption="Your photo")

with result_column:
    if best_score >= CONFIDENCE_LIMIT:
        st.success(f"This looks like **{DISPLAY_NAMES[best_name]}** ({best_score:.0f}%).")
    else:
        st.warning(
            f"The model is unsure. Its best guess is **{DISPLAY_NAMES[best_name]}** "
            f"({best_score:.0f}%). The photo may show something outside the nine classes, "
            "or the background and lighting may differ from the training photos."
        )

    st.subheader("Top three")
    for name, score in ranking.head(3).items():
        st.write(f"{DISPLAY_NAMES[name]} – {score:.1f}%")
        st.progress(min(max(score / 100, 0.0), 1.0))

st.subheader("All nine classes (%)")
st.bar_chart(ranking.rename(index=DISPLAY_NAMES))

# --- Nutritional information ---------------------------------------------------
st.divider()
st.subheader("Nutritional information")
st.info("Coming next: nutritional values from Livsmedelsverket for the predicted food.")

# --- About the model -------------------------------------------------------------
with st.expander("About the model"):
    st.write(
        "The app uses a convolutional neural network (CNN) trained on 1,603 photos. "
        "On 200 test photos from the same dataset it classified 98.5% correctly. "
        "All dataset photos were taken in a few photography sessions, so the model can "
        "perform worse on photos with other backgrounds or lighting. It can only choose "
        "between the nine classes above."
    )
