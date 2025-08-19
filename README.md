# Sms_spam_or_ham_detection-
Streamlit app that classifies email/SMS as Spam or Not Spam using a pre-trained scikit-learn model. It lowercases, tokenizes, removes punctuation and stopwords (with sklearn fallback), stems, then applies a saved TF‑IDF vectorizer (vectorizer.pkl) and predicts with a saved model (model.pkl).
