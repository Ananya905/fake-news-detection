import streamlit as st
import pickle
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection App",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection App")
st.write("Enter a news article below to check whether it is **Fake** or **Real**.")

# -----------------------------
# Load model & vectorizer safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
tfidf_path = os.path.join(BASE_DIR, "tfidf.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(tfidf_path, "rb") as f:
    tfidf = pickle.load(f)

# -----------------------------
# User input
# -----------------------------
news_text = st.text_area(
    "Enter news article",
    height=180,
    placeholder="Paste the full news article here..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter a news article before clicking Predict.")
    else:
        # Transform input
        vectorized_text = tfidf.transform([news_text])

        # Predict
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0].max()

        # Show result
        if prediction == 1:
            st.error("🛑 Prediction: **FAKE NEWS**")
        else:
            st.success("✅ Prediction: **REAL NEWS**")

        st.write(f"**Confidence:** {probability:.2f}")
