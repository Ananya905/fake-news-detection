import streamlit as st
from transformers import pipeline

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
# Load pre-trained fake news detection pipeline
# -----------------------------
@st.cache_resource
def load_model():
    # Using a Hugging Face model fine-tuned for fake news detection
    # You can also replace with another model from Hugging Face if desired
    classifier = pipeline("text-classification", model="mrm8488/bert-mini-finetuned-fake-news-detection")
    return classifier

classifier = load_model()

# -----------------------------
# User input
# -----------------------------
news_text = st.text_area(
    "Enter news article",
    height=200,
    placeholder="Paste the full news article here..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    if not news_text.strip():
        st.warning("⚠️ Please enter a news article before clicking Predict.")
    else:
        result = classifier(news_text[:512])[0]  # Truncate if long text
        label = result['label']
        score = result['score']

        if label.lower() == "fake":
            st.error(f"🛑 Prediction: **FAKE NEWS**")
        else:
            st.success(f"✅ Prediction: **REAL NEWS**")
        
        st.write(f"**Confidence:** {score:.2f}")
