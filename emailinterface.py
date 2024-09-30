import streamlit as st
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained model and vectorizer
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def pre_process(mail):
    # Convert to lowercase
    mail = mail.lower()
    
    # Remove punctuation and non-alphabetic characters using regex
    mail = re.sub(r'[^\w\s]', '', mail)
    mail = re.sub(r'[^a-zA-Z\s]', '', mail)
    
    # Tokenize (split the text into words)
    mail = mail.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    mail = [word for word in mail if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    mail = [lemmatizer.lemmatize(word) for word in mail]
    
    # Rejoin the tokens into a single string
    mail = ' '.join(mail)
    
    return mail

def predict_spam(email_text):
    # Preprocess the email text
    processed_mail = pre_process(email_text)

    # Transform the mail using the already fitted vectorizer
    mail_vector = vectorizer.transform([processed_mail])

    # Predict the label using the trained model
    prediction = model.predict(mail_vector)
    
    return prediction[0]  # Return the prediction

# Streamlit interface
st.title("Email Spam Detection")
st.markdown("""
    This application uses a machine learning model to classify emails as Spam or Not Spam.
    Please enter the email content below and click on 'Check Spam'.
""")

email_text = st.text_area("Enter the email text:", height=300)

if st.button("Check Spam"):
    if email_text:
        with st.spinner("Checking..."):
            prediction = predict_spam(email_text)
            result = "Spam" if prediction == 1 else "Not Spam"
            st.success(f"The email is: **{result}**")
    else:
        st.error("Please enter some email text.")

# Add some custom styling
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        font-size: 16px;
        padding: 10px;
        border: none;
        border-radius: 5px;
    }
    .stTextArea > div > textarea {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)