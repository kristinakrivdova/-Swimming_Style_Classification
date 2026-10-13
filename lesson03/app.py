"""Streamlit application for swimming-style classification using fastai."""

import pathlib
import sys

import streamlit as st
from fastai.vision.all import PILImage, load_learner
from huggingface_hub import hf_hub_download

# Compatibility with models exported from Google Colab.
sys.modules.setdefault("pathlib._local", pathlib)

REPO_ID = "Kryskriv/Swimming_Style_Classification"

SLOVAK_NAMES = {
    "backstroke": "Znak",
    "breaststroke": "Prsia",
    "butterfly": "Motýlik",
    "front crawl": "Kraul",
}

st.set_page_config(page_title="Swimming Style Classification", page_icon="🏊", layout="centered")

st.title("🏊 Swimming Style Classification")
st.write(
    "Upload a swimming photo and let the model predict one of four swimming styles: "
    "butterfly, backstroke, breaststroke, or front crawl."
)


@st.cache_resource(show_spinner="Loading the swimming classifier...")
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pkl")
    return load_learner(model_path, cpu=True)


try:
    learn = load_model()
except Exception as exc:
    st.error("The model could not be loaded from Hugging Face.")
    st.exception(exc)
    st.stop()

uploaded = st.file_uploader("Choose a swimming photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded photo", use_container_width=True)
    prediction, _, probabilities = learn.predict(PILImage.create(uploaded.getvalue()))

    predicted_label = str(prediction)
    translated_label = SLOVAK_NAMES.get(predicted_label.lower().strip(), predicted_label)
    st.subheader(f"Prediction: {predicted_label} ({translated_label})")

    st.write("**Model confidence by swimming style**")
    class_probabilities = sorted(
        zip(learn.dls.vocab, probabilities),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    for category, probability in class_probabilities:
        confidence = float(probability)
        st.progress(confidence, text=f"{category}: {confidence:.1%}")

    st.caption(
        "The values are model probabilities, not a guarantee of correctness. "
        "Unusual camera angles and similar phases of strokes can cause errors."
    )

st.divider()
st.caption("Fastai image classifier · Model hosted on Hugging Face")
