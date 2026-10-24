"""
streamlit_app.py - a small web app to try ReviewLens on any review.

What it does
------------
1. You paste (or pick) a product review.
2. The app shows the OVERALL sentiment (positive / neutral / negative) with the
   model's confidence for each class.
3. It also finds the ASPECTS mentioned (battery, camera, price, ...) and shows
   the sentiment of each one separately.

Which model does it use?
------------------------
The Logistic Regression model saved by notebook 02 (models/baseline_tfidf_lr.pkl).
We use it because it can output a probability for each class, which lets the app
show a confidence. (On unseen texts it performs about the same as the SVM.)

Run it (from the project root, with the virtual environment active):
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Make "from src... import" work no matter where the app is started from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.aspects import extract_aspect_clauses
from src.preprocess import clean_text

MODEL_PATH = ROOT / "models" / "baseline_tfidf_lr.pkl"
ORDER = ["positive", "neutral", "negative"]
EMOJI = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}

EXAMPLES = {
    "Write my own": "",
    "Mixed review": "Camera is great but battery is bad",
    "Positive": "Excellent sound quality and value for money, delivery was fast",
    "Negative": "Stopped working after a week. Waste of money, not recommended",
    "Negation": "Not good at all, the screen is very dull",
}

st.set_page_config(page_title="ReviewLens", page_icon="🔍")


@st.cache_resource
def load_model():
    """Load the vectorizer and classifier once and reuse them."""
    return joblib.load(MODEL_PATH)


def predict_probabilities(vec, clf, text):
    """Return {class: probability} for a piece of text, or None if it has no usable words."""
    cleaned = clean_text(text)
    if not cleaned.strip():
        return None
    probabilities = clf.predict_proba(vec.transform([cleaned]))[0]
    return dict(zip(clf.classes_, probabilities))


# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "ReviewLens predicts the sentiment of a product review and finds which "
        "aspects (battery, camera, price, ...) customers talk about."
    )
    st.write(
        "Model: TF-IDF + Logistic Regression, trained on more than 67,000 "
        "Flipkart electronics reviews."
    )
    st.caption(
        "Positive and negative predictions are fairly reliable. Neutral is the "
        "hardest class for every model we tried (F1 about 0.3), so treat neutral "
        "predictions with care."
    )

# ---------- Main page ----------
st.title("🔍 ReviewLens")
st.write("Paste an electronics product review to see its sentiment and the aspects it mentions.")

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run notebook `02_baseline_model.ipynb` first; "
        "it saves `models/baseline_tfidf_lr.pkl`."
    )
    st.stop()

vec, clf = load_model()

example = st.selectbox("Try an example", list(EXAMPLES))
review = st.text_area("Review text", value=EXAMPLES[example], height=120)

if st.button("Analyze", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
        st.stop()

    overall = predict_probabilities(vec, clf, review)
    if overall is None:
        st.warning("There is not enough text to analyze (after cleaning, nothing is left).")
        st.stop()

    # --- Overall sentiment ---
    label = max(overall, key=overall.get)
    st.subheader(f"Overall sentiment: {EMOJI[label]} {label.title()}")
    st.caption(f"Confidence: {overall[label]:.0%}")
    for cls in ORDER:
        st.write(f"{EMOJI[cls]} {cls.title()}: {overall[cls]:.0%}")
        st.progress(float(overall[cls]))

    # --- Aspect-level sentiment ---
    st.subheader("Aspects mentioned")
    rows = []
    for aspect, clause in extract_aspect_clauses(review):
        probabilities = predict_probabilities(vec, clf, clause)
        if probabilities is None or len(clean_text(clause).split()) < 2:
            continue  # a single word carries no sentiment
        aspect_label = max(probabilities, key=probabilities.get)
        rows.append({
            "Aspect": aspect,
            "Clause": clause,
            "Sentiment": f"{EMOJI[aspect_label]} {aspect_label.title()}",
            "Confidence": f"{probabilities[aspect_label]:.0%}",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("No specific aspect was found in this review (it reads as a general comment).")
