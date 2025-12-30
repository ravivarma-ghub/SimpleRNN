import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# =========================
# CONSTANTS (MUST MATCH TRAINING)
# =========================
MAX_WORDS = 10000   # same as num_words used during training
MAX_LEN = 500       # same maxlen used during training

# =========================
# LOAD WORD INDEX
# =========================
word_index = imdb.get_word_index()

# =========================
# LOAD TRAINED MODEL
# =========================
model = load_model("simple_rnn_imdb.h5")

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_review(text):
    words = text.lower().split()
    encoded_review = []

    for word in words:
        index = word_index.get(word)
        if index is not None and index < MAX_WORDS:
            encoded_review.append(index)
        else:
            encoded_review.append(2)  # unknown word token

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=MAX_LEN
    )
    return padded_review

# =========================
# PREDICTION FUNCTION
# =========================
def predict_sentiment(review):
    processed_review = preprocess_review(review)
    prediction = model.predict(processed_review, verbose=0)[0][0]

    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return sentiment, prediction

# =========================
# STREAMLIT APP
# =========================
st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write("Enter a movie review below to predict whether the sentiment is **Positive** or **Negative**.")

user_input = st.text_area("Movie Review")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.success(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.4f}")
else:
    st.info("Type a review and click **Predict Sentiment**.")
