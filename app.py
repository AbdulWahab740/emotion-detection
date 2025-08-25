import streamlit as st
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
# Make sure you download these if not already
import pickle
import json

model = tf.keras.models.load_model("lstm_model Emotion Detection.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.json") as f:
    label_map = json.load(f)

MAX_LEN = 30

# -------------------------
# Text Preprocessing Functions
# -------------------------
stop_words = set(stopwords.words("english"))
stop_words.discard("not")

def lower_case(text):
    return " ".join(word.lower() for word in text.split())

def remove_numbers(text):
    return ''.join(char for char in text if not char.isdigit())

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def normalize_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_numbers(sentence)
    sentence = remove_urls(sentence)
    return sentence

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Emotion Detection", page_icon="🙂")

st.title("🙂 Emotion Detection Sentiment Analysis")
st.write("A Deep learning model which tells you about the emotion in the sentence whether it is postive, negative or neutral")

st.markdown("Write your sentence below to see the sentiment of the sentence.")

# Input Box
user_input = st.text_area("✍️ Enter your sentence", height=150)

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence first.")
    else:
        user_input = normalize_sentence(user_input)
        seq = tokenizer.texts_to_sequences([user_input])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
        probs = model.predict(padded)[0]
        class_idx = np.argmax(probs)
        st.success(f"Prediction: {label_map[class_idx]}")
        st.write(f"Confidence: {probs[class_idx]:.2f}")
