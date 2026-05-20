import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import numpy as np
import easyocr
import os
import requests
import pyrebase
import firebase_admin

from PIL import Image
from newspaper import Article
from newspaper.article import ArticleException

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from firebase_admin import credentials
from firebase_admin import firestore

from datetime import datetime

# ==================================================
# NLTK
# ==================================================

nltk.download("stopwords")
nltk.download("wordnet")

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# ==================================================
# FIREBASE CONFIG
# ==================================================

firebase_config = {

    "apiKey":
    st.secrets["firebase"]["apiKey"],

    "authDomain":
    st.secrets["firebase"]["authDomain"],

    "projectId":
    st.secrets["firebase"]["projectId"],

    "storageBucket":
    st.secrets["firebase"]["storageBucket"],

    "messagingSenderId":
    st.secrets["firebase"]["messagingSenderId"],

    "appId":
    st.secrets["firebase"]["appId"],

    "databaseURL":
    ""

}

firebase = pyrebase.initialize_app(
    firebase_config
)

auth = firebase.auth()

# ==================================================
# FIREBASE ADMIN
# ==================================================

if not firebase_admin._apps:

    cred = credentials.Certificate({

        "type":
        st.secrets["firebase_service_account"]["type"],

        "project_id":
        st.secrets["firebase_service_account"]["project_id"],

        "private_key_id":
        st.secrets["firebase_service_account"]["private_key_id"],

        "private_key":
        st.secrets["firebase_service_account"]["private_key"],

        "client_email":
        st.secrets["firebase_service_account"]["client_email"],

        "client_id":
        st.secrets["firebase_service_account"]["client_id"],

        "auth_uri":
        st.secrets["firebase_service_account"]["auth_uri"],

        "token_uri":
        st.secrets["firebase_service_account"]["token_uri"],

        "auth_provider_x509_cert_url":
        st.secrets["firebase_service_account"]["auth_provider_x509_cert_url"],

        "client_x509_cert_url":
        st.secrets["firebase_service_account"]["client_x509_cert_url"]

    })

    firebase_admin.initialize_app(cred)

db = firestore.client()

# ==================================================
# SESSION STATE
# ==================================================

if "admin_logged_in" not in st.session_state:

    st.session_state.admin_logged_in = False

# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load(
    "models/fake_news_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

# ==================================================
# OCR
# ==================================================

reader = easyocr.Reader(['en'])

# ==================================================
# STOPWORDS
# ==================================================

stop_words = set(
    stopwords.words("english")
)

custom_stopwords = {

    "said",
    "reuters",
    "trump",
    "donald",
    "president",
    "u",
    "us",
    "say",
    "would",
    "could",
    "also",
    "one",
    "may",
    "even",
    "people",
    "told",
    "new",
    "time"

}

stop_words.update(
    custom_stopwords
)

# ==================================================
# LEMMATIZER
# ==================================================

lemmatizer = WordNetLemmatizer()

# ==================================================
# CLEAN TEXT
# ==================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"www\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

# ==================================================
# REMOVE STOPWORDS
# ==================================================

def remove_stopwords(text):

    words = text.split()

    filtered_words = [

        word for word in words

        if word not in stop_words

    ]

    return " ".join(
        filtered_words
    )

# ==================================================
# LEMMATIZATION
# ==================================================

def lemmatize_text(text):

    words = text.split()

    lemmatized_words = [

        lemmatizer.lemmatize(word)

        for word in words

    ]

    return " ".join(
        lemmatized_words
    )

# ==================================================
# URL EXTRACTION
# ==================================================

def extract_news_from_url(url):

    try:

        headers = {

            "User-Agent":
            "Mozilla/5.0"

        }

        response = requests.get(

            url,

            headers=headers,

            timeout=10

        )

        article = Article(url)

        article.set_html(
            response.text
        )

        article.parse()

        return article.text

    except ArticleException:

        return ""

    except Exception:

        return ""

# ==================================================
# OCR IMAGE
# ==================================================

def extract_text_from_image(image):

    result = reader.readtext(

        np.array(image),

        detail=0

    )

    return " ".join(result)

# ==================================================
# TITLE
# ==================================================

st.title(
    "📰 Fake News Detection System"
)

# ==================================================
# DESCRIPTION
# ==================================================

st.write(
    """
    Analyze news articles using:

    • News URL  
    • News screenshots/images

    AI predicts whether the news is Fake or Real.
    """
)

# ==================================================
# ADMIN LOGIN
# ==================================================

st.sidebar.title(
    "🔐 Admin Login"
)

admin_email = st.sidebar.text_input(
    "Admin Email"
)

admin_password = st.sidebar.text_input(
    "Admin Password",
    type="password"
)

if st.sidebar.button(
    "Login"
):

    try:

        auth.sign_in_with_email_and_password(

            admin_email,
            admin_password

        )

        st.session_state.admin_logged_in = True

        st.sidebar.success(
            "Login successful"
        )

        st.rerun()

    except Exception:

        st.sidebar.error(
            "Invalid email or password"
        )

# ==================================================
# ADMIN PANEL
# ==================================================

if st.session_state.admin_logged_in:

    st.sidebar.success(
        "Admin Access Granted"
    )

    # ==============================================
    # ADD TRAINING DATA
    # ==============================================

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "Add New Training Data"
    )

    admin_url = st.sidebar.text_input(
        "News URL"
    )

    admin_image = st.sidebar.file_uploader(

        "Upload News Image",

        type=[
            "png",
            "jpg",
            "jpeg"
        ],

        key="admin_image"

    )

    admin_text = st.sidebar.text_area(
        "Or Paste News Text"
    )

    admin_label = st.sidebar.selectbox(

        "Label",

        [
            "Fake",
            "Real"
        ]

    )

    if st.sidebar.button(
        "Save Training Data"
    ):

        try:

            final_text = ""

            if admin_url.strip() != "":

                final_text = extract_news_from_url(
                    admin_url
                )

            elif admin_image is not None:

                image = Image.open(
                    admin_image
                )

                final_text = extract_text_from_image(
                    image
                )

            elif admin_text.strip() != "":

                final_text = admin_text

            else:

                st.sidebar.warning(
                    "Please provide URL, image or text."
                )

            if final_text != "":

                cleaned_text = clean_text(
                    final_text
                )

                cleaned_text = remove_stopwords(
                    cleaned_text
                )

                cleaned_text = lemmatize_text(
                    cleaned_text
                )

                try:

                    existing_data = pd.read_csv(
                        "../data/admin_data.csv"
                    )

                except:

                    existing_data = pd.DataFrame(

                        columns=[
                            "text",
                            "clean_text",
                            "label"
                        ]

                    )

                if cleaned_text in existing_data[
                    "clean_text"
                ].values:

                    st.sidebar.warning(
                        "This news already exists."
                    )

                else:

                    label_value = (

                        0 if admin_label == "Fake"

                        else 1

                    )

                    new_data = pd.DataFrame({

                        "text":
                        [final_text],

                        "clean_text":
                        [cleaned_text],

                        "label":
                        [label_value]

                    })

                    updated_data = pd.concat(

                        [
                            existing_data,
                            new_data
                        ],

                        ignore_index=True

                    )

                    updated_data.to_csv(

                        "data/admin_data.csv",

                        index=False

                    )

                    st.sidebar.success(
                        "Training data saved."
                    )

                    st.sidebar.info(
                        "Retraining model..."
                    )

                    os.system(
                        "python train.py"
                    )

                    model = joblib.load(
                        "models/fake_news_model.pkl"
                    )

                    vectorizer = joblib.load(
                        "models/tfidf_vectorizer.pkl"
                    )

                    st.sidebar.success(
                        "Model retrained."
                    )

        except Exception as e:

            st.sidebar.error(
                "Could not save training data."
            )

            st.sidebar.write(e)

    # ==============================================
    # MANAGE SAVED NEWS
    # ==============================================

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "Manage Saved News"
    )

    try:

        admin_dataset = pd.read_csv(
            "data/admin_data.csv"
        )

        if len(admin_dataset) > 0:

            selected_index = st.sidebar.selectbox(

                "Select News",

                options=admin_dataset.index,

                format_func=lambda x:
                admin_dataset.loc[x, "text"][:80] + "..."

            )

            selected_row = admin_dataset.loc[
                selected_index
            ]

            st.sidebar.write(
                selected_row["text"][:500]
            )

            current_label = (

                "Fake"

                if selected_row["label"] == 0

                else "Real"

            )

            st.sidebar.write(
                f"Current Label: {current_label}"
            )

            new_label = st.sidebar.selectbox(

                "Update Label",

                [
                    "Fake",
                    "Real"
                ],

                key="update_label"

            )

            if st.sidebar.button(
                "Update News Label"
            ):

                admin_dataset.loc[
                    selected_index,
                    "label"
                ] = (

                    0 if new_label == "Fake"
                    else 1

                )

                admin_dataset.to_csv(

                    "data/admin_data.csv",

                    index=False

                )

                st.sidebar.success(
                    "Label updated."
                )

                os.system(
                    "python train.py"
                )

                st.rerun()

            if st.sidebar.button(
                "Delete Selected News"
            ):

                admin_dataset = admin_dataset.drop(
                    selected_index
                )

                admin_dataset.reset_index(

                    drop=True,

                    inplace=True

                )

                admin_dataset.to_csv(

                    "data/admin_data.csv",

                    index=False

                )

                st.sidebar.success(
                    "News deleted."
                )

                os.system(
                    "python train.py"
                )

                st.rerun()

        else:

            st.sidebar.info(
                "No saved news."
            )

    except Exception as e:

        st.sidebar.error(
            "Could not load dataset."
        )

        st.sidebar.write(e)

    # ==============================================
    # FEEDBACK PANEL
    # ==============================================

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "📩 User Feedbacks"
    )

    feedback_docs = db.collection(
        "feedbacks"
    ).stream()

    feedback_list = []

    for doc in feedback_docs:

        feedback_data = doc.to_dict()

        feedback_data["doc_id"] = doc.id

        feedback_list.append(
            feedback_data
        )

    # ==========================================
    # SEARCH
    # ==========================================

    search_feedback = st.sidebar.text_input(
        "🔍 Search Feedback"
    )

    # ==========================================
    # SORT
    # ==========================================

    sort_option = st.sidebar.selectbox(

        "Sort Feedbacks",

        [
            "Newest First",
            "Oldest First"
        ]

    )

    # ==========================================
    # SEARCH FILTER
    # ==========================================

    if search_feedback.strip() != "":

        feedback_list = [

            feedback for feedback in feedback_list

            if search_feedback.lower()

            in feedback.get(
                "name",
                ""
            ).lower()

            or

            search_feedback.lower()

            in feedback.get(
                "feedback",
                ""
            ).lower()

        ]

    # ==========================================
    # SORTING
    # ==========================================

    if sort_option == "Newest First":

        feedback_list = sorted(

            feedback_list,

            key=lambda x:
            x.get(
                "timestamp",
                datetime.min
            ),

            reverse=True

        )

    else:

        feedback_list = sorted(

            feedback_list,

            key=lambda x:
            x.get(
                "timestamp",
                datetime.min
            )

        )

    # ==========================================
    # SHOW FEEDBACKS
    # ==========================================

    if len(feedback_list) > 0:

        for feedback in feedback_list[:20]:

            with st.sidebar.expander(

                f"👤 {feedback['name']}",

                expanded=False

            ):

                st.write(
                    feedback["feedback"]
                )

                timestamp = feedback.get(
                    "timestamp"
                )

                if timestamp:

                    formatted_time = timestamp.strftime(
                        "%d.%m.%Y %H:%M"
                    )

                    st.caption(
                        f"🕒 {formatted_time}"
                    )

                if st.button(

                    "Delete Feedback",

                    key=feedback["doc_id"]

                ):

                    db.collection(
                        "feedbacks"
                    ).document(
                        feedback["doc_id"]
                    ).delete()

                    st.success(
                        "Feedback deleted"
                    )

                    st.rerun()

    else:

        st.sidebar.info(
            "No feedback found."
        )

# ==================================================
# USER PANEL
# ==================================================

st.header(
    "Analyze News"
)

col1, col2 = st.columns(2)

with col1:

    news_url = st.text_input(
        "Paste News URL Here"
    )

with col2:

    uploaded_image = st.file_uploader(

        "Upload News Image",

        type=[
            "png",
            "jpg",
            "jpeg"
        ]

    )

# ==================================================
# ANALYZE BUTTON
# ==================================================

if st.button(
    "Analyze News"
):

    try:

        input_text = ""

        if news_url.strip() != "":

            with st.spinner(
                "Extracting article..."
            ):

                input_text = extract_news_from_url(
                    news_url
                )

        elif uploaded_image is not None:

            with st.spinner(
                "Reading image..."
            ):

                image = Image.open(
                    uploaded_image
                )

                input_text = extract_text_from_image(
                    image
                )

        else:

            st.warning(
                "Please provide URL or image."
            )

        if input_text != "":

            cleaned_text = clean_text(
                input_text
            )

            cleaned_text = remove_stopwords(
                cleaned_text
            )

            cleaned_text = lemmatize_text(
                cleaned_text
            )

            vectorized_text = vectorizer.transform(
                [cleaned_text]
            )

            prediction = model.predict(
                vectorized_text
            )[0]

            confidence = model.decision_function(
                vectorized_text
            )[0]

            st.subheader(
                "Prediction Result"
            )

            st.write(
                f"Confidence Score: {confidence:.2f}"
            )

            if prediction == 0:

                st.error(
                    "🚨 Fake News"
                )

            else:

                st.success(
                    "✅ Real News"
                )

    except Exception as e:

        st.error(
            "Could not analyze news."
        )

        st.write(e)

# ==================================================
# FEEDBACK SECTION
# ==================================================

st.markdown("---")

st.header(
    "💬 Feedback"
)

feedback_name = st.text_input(
    "Your Name"
)

feedback_text = st.text_area(
    "Write your feedback"
)

if st.button(
    "Send Feedback"
):

    if feedback_name.strip() == "" \
    or feedback_text.strip() == "":

        st.warning(
            "Please fill all fields."
        )

    else:

        db.collection(
            "feedbacks"
        ).add({

            "name":
            feedback_name,

            "feedback":
            feedback_text,

            "timestamp":
            datetime.utcnow()

        })

        st.success(
            "Feedback sent successfully."
        )