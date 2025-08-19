import streamlit as st
import pickle
import string
import re
import nltk
from pathlib import Path
from nltk.stem.porter import PorterStemmer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOP_WORDS

ps = PorterStemmer()
 
def ensure_nltk_resources() -> None:
    """Attempt to ensure tokenizer and stopwords resources exist.
    Tries both legacy and new resource names and degrades silently if downloads fail.
    """
    # punkt (legacy) and punkt_tab (newer NLTK builds)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass
    # stopwords corpus
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass


# Ensure necessary NLTK resources are available at startup (best-effort)
ensure_nltk_resources()
try:
    from nltk.corpus import stopwords as nltk_stopwords
    stop_words = set(nltk_stopwords.words('english'))
except Exception:
    # Fallback in case NLTK stopwords loader has issues
    stop_words = set(SKLEARN_STOP_WORDS)


def transform_text(text):
    text = text.lower()
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        # Try to fetch missing resources and retry once
        ensure_nltk_resources()
        try:
            tokens = nltk.word_tokenize(text)
        except Exception:
            # Final fallback: simple regex-based tokenization
            tokens = re.findall(r"\b\w+\b", text)
    text = tokens

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load artifacts relative to this file location (works locally and on Streamlit Cloud)
BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / 'vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open(BASE_DIR / 'model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    try:
        # Ensure vectorizer and model are fitted
        if not hasattr(tfidf, "idf_"):
            st.error("Your TF-IDF vectorizer is not fitted. Re-train and re-save `vectorizer.pkl`.")
        else:
            check_is_fitted(model)
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            # 4. Display
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
    except NotFittedError:
        st.error("Your model or vectorizer is not fitted. Please re-train and re-save the fitted artifacts.")



        
