import streamlit as st
import joblib

# --- Load Saved Files ---
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# --- Clean text function ---
def clean_text(text):
    return text.lower()

# --- Page Config ---
st.set_page_config(page_title="Spam Detector", page_icon="✉️", layout="centered")

# --- Custom CSS ---
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.05);
    color: white;
    border-radius: 15px;
    font-size: 16px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(5px);
}
.stButton>button {
    background: linear-gradient(135deg, #ff512f, #dd2476);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 2rem;
    font-size: 16px;
    font-weight: bold;
    border: none;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}
.result-box {
    padding: 20px;
    border-radius: 20px;
    font-size: 18px;
    margin-top: 25px;
    text-align: center;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}
.spam {
    background: rgba(255, 76, 76, 0.7);
    color: white;
    font-weight: bold;
}
.not-spam {
    background: rgba(46, 125, 50, 0.7);
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align:center; color:white;'>🚀 Email Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Check your messages in seconds ⚡</p>", unsafe_allow_html=True)

# --- Input area ---
user_input = st.text_area("✉️ Enter your message here:")

# --- Prediction ---
if st.button("🔍 Predict"):
    if user_input.strip():
        cleaned_message = clean_text(user_input)
        vectorized_message = vectorizer.transform([cleaned_message])
        prediction = model.predict(vectorized_message)[0]

        if prediction == 1:
            st.markdown("<div class='result-box spam'>🚫 This is a Spam message.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box not-spam'>✅ This is Not Spam.</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message to check.")
