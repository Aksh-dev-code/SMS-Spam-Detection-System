import streamlit as st
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#  Page Config 
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

#Css
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stTextInput>div>div>input {
    background-color: #262730;
    color: white;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.spam {
    background-color: #ff4b4b;
    color: white;
}
.not-spam {
    background-color: #00c853;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# NLP Setup
@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

load_nltk()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def text_preproceser(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Load Mode
tfidf = pickle.load(open('artifacts/vectorizer.pkl','rb'))
model = pickle.load(open('artifacts/model.pkl','rb'))

# UI
st.title("📩 SMS Spam Detection")
st.markdown("Detect whether a message is **Spam or Not Spam** using Machine Learning 🤖")

input_sms = st.text_area("✉️ Enter your message here", height=150)

if st.button("🔍 Analyze Message"):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        transformed_sms = text_preproceser(input_sms)
        vector_input = tfidf.transform([transformed_sms]).toarray()

        prediction = model.predict(vector_input)[0]

        # Confidence (if available)
        try:
            prob = model.predict_proba(vector_input)[0]
            confidence = max(prob) * 100
        except:
            confidence = None

        st.markdown("### 📊 Result")

        if prediction == 1:
            st.markdown(
                f'<div class="result-box spam">🚨 SPAM MESSAGE</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box not-spam">✅ NOT SPAM</div>',
                unsafe_allow_html=True
            )

        if confidence:
            st.info(f"Confidence: {confidence:.2f}%")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built using Streamlit | NLP Project")