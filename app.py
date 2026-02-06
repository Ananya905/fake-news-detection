import streamlit as st
import pickle
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# Load trained model and TF-IDF vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter news article")

if st.button("Predict"):
    cleaned_text = clean_text(user_input)
    vector = tfidf.transform([cleaned_text])
    prediction = model.predict(vector)[0]
    st.success(f"Prediction: {prediction}")
