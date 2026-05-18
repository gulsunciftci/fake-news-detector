import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import numpy as np
import easyocr
import os

from PIL import Image
from newspaper import Article

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==================================================
# DOWNLOAD NLTK DATA
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
# LOAD MODEL
# ==================================================

model = joblib.load(
    "models/fake_news_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

# ==================================================
# OCR READER
# ==================================================

reader = easyocr.Reader(['en'])

# ==================================================
# STOPWORDS
# ==================================================

stop_words = set(stopwords.words("english"))

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

stop_words.update(custom_stopwords)

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

    return " ".join(filtered_words)

# ==================================================
# LEMMATIZATION
# ==================================================

def lemmatize_text(text):

    words = text.split()

    lemmatized_words = [

        lemmatizer.lemmatize(word)

        for word in words

    ]

    return " ".join(lemmatized_words)

# ==================================================
# EXTRACT NEWS FROM URL
# ==================================================

def extract_news_from_url(url):

    article = Article(url)

    article.download()

    article.parse()

    return article.text

# ==================================================
# OCR FROM IMAGE
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

st.title("📰 Fake News Detection System")

# ==================================================
# USER PANEL
# ==================================================

st.header("User Panel")

st.write(
    """
    Analyze news articles using:

    • News URL  
    • News screenshots/images

    The AI model predicts whether
    the news is Fake or Real.
    """
)

# ==================================================
# ADMIN PANEL
# ==================================================

st.sidebar.title("🔐 Admin Panel")

admin_password = st.sidebar.text_input(
    "Admin Password",
    type="password"
)

if admin_password == "myadminpane123":

    st.sidebar.success(
        "Admin Access Granted"
    )

    st.sidebar.subheader(
        "Add New Training Data"
    )

    # ==============================================
    # ADMIN URL INPUT
    # ==============================================

    admin_url = st.sidebar.text_input(
        "News URL"
    )

    # ==============================================
    # ADMIN IMAGE INPUT
    # ==============================================

    admin_image = st.sidebar.file_uploader(
        "Upload News Image",
        type=["png", "jpg", "jpeg"],
        key="admin_image"
    )

    # ==============================================
    # ADMIN TEXT INPUT
    # ==============================================

    admin_text = st.sidebar.text_area(
        "Or Paste News Text"
    )

    # ==============================================
    # LABEL
    # ==============================================

    admin_label = st.sidebar.selectbox(
        "Label",
        ["Fake", "Real"]
    )

    # ==============================================
    # SAVE BUTTON
    # ==============================================

    if st.sidebar.button(
        "Save Training Data"
    ):

        try:

            final_text = ""

            # ======================================
            # URL INPUT
            # ======================================

            if admin_url.strip() != "":

                extracted_url_text = (
                    extract_news_from_url(
                        admin_url
                    )
                )

                final_text = extracted_url_text

            # ======================================
            # IMAGE INPUT
            # ======================================

            elif admin_image is not None:

                image = Image.open(
                    admin_image
                )

                extracted_image_text = (
                    extract_text_from_image(
                        image
                    )
                )

                final_text = extracted_image_text

            # ======================================
            # TEXT INPUT
            # ======================================

            elif admin_text.strip() != "":

                final_text = admin_text

            # ======================================
            # EMPTY INPUT
            # ======================================

            else:

                st.sidebar.warning(
                    "Please provide URL, image or text."
                )

            # ======================================
            # CONTINUE IF TEXT EXISTS
            # ======================================

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

                # ==================================
                # LOAD EXISTING DATA
                # ==================================

                try:

                    existing_data = pd.read_csv(
                        "data/admin_data.csv"
                    )

                except:

                    existing_data = pd.DataFrame(
                        columns=[
                            "text",
                            "clean_text",
                            "label"
                        ]
                    )

                # ==================================
                # DUPLICATE CHECK
                # ==================================

                if cleaned_text in existing_data[
                    "clean_text"
                ].values:

                    st.sidebar.warning(
                        "This news already exists in admin_data.csv"
                    )

                else:

                    # ==============================
                    # LABEL VALUE
                    # ==============================

                    label_value = (
                        0 if admin_label == "Fake"
                        else 1
                    )

                    # ==============================
                    # NEW DATAFRAME
                    # ==============================

                    new_data = pd.DataFrame({

                        "text": [final_text],

                        "clean_text": [cleaned_text],

                        "label": [label_value]

                    })

                    # ==============================
                    # CONCAT DATA
                    # ==============================

                    updated_data = pd.concat(

                        [existing_data, new_data],

                        ignore_index=True

                    )

                    # ==============================
                    # SAVE DATA
                    # ==============================

                    updated_data.to_csv(

                        "data/admin_data.csv",

                        index=False

                    )

                    st.sidebar.success(
                        "Training data saved successfully."
                    )

                    st.sidebar.write(
                        f"Dataset Size: {len(updated_data)}"
                    )

                    # ==============================
                    # AUTO RETRAIN
                    # ==============================

                    st.sidebar.info(
                        "Retraining model..."
                    )

                    os.system(
                        "python train.py"
                    )

                    # ==============================
                    # RELOAD MODEL
                    # ==============================

                    model = joblib.load(
                        "models/fake_news_model.pkl"
                    )

                    vectorizer = joblib.load(
                        "models/tfidf_vectorizer.pkl"
                    )

                    st.sidebar.success(
                        "Model retrained successfully."
                    )

        except Exception as e:

            st.sidebar.error(
                "Could not save training data."
            )

            st.sidebar.write(e)

    # ==================================================
    # ADMIN DATA MANAGEMENT
    # ==================================================

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "Manage Saved News"
    )

    try:

        admin_dataset = pd.read_csv(
            "data/admin_data.csv"
        )

        if len(admin_dataset) > 0:

            # ======================================
            # SELECT NEWS
            # ======================================

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
                "### Selected News"
            )

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

            # ======================================
            # UPDATE LABEL
            # ======================================

            new_label = st.sidebar.selectbox(

                "Update Label",

                ["Fake", "Real"],

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
                    "News label updated successfully."
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
                    "Model retrained successfully."
                )

            # ======================================
            # DELETE NEWS
            # ======================================

            st.sidebar.markdown("---")

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
                    "News deleted successfully."
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
                    "Model retrained successfully."
                )

        else:

            st.sidebar.info(
                "No saved admin news found."
            )

    except Exception as e:

        st.sidebar.error(
            "Could not load admin dataset."
        )

        st.sidebar.write(e)

# ==================================================
# USER INPUTS
# ==================================================

col1, col2 = st.columns(2)

with col1:

    news_url = st.text_input(
        "Paste News URL Here"
    )

with col2:

    uploaded_image = st.file_uploader(
        "Upload News Image",
        type=["png", "jpg", "jpeg"]
    )

# ==================================================
# ANALYZE BUTTON
# ==================================================

if st.button("Analyze News"):

    try:

        input_text = ""

        # ==========================================
        # URL INPUT
        # ==========================================

        if news_url.strip() != "":

            with st.spinner(
                "Extracting article from URL..."
            ):

                article_text = extract_news_from_url(
                    news_url
                )

                input_text = article_text

        # ==========================================
        # IMAGE INPUT
        # ==========================================

        elif uploaded_image is not None:

            with st.spinner(
                "Reading text from image..."
            ):

                image = Image.open(
                    uploaded_image
                )

                extracted_text = extract_text_from_image(
                    image
                )

                input_text = extracted_text

        # ==========================================
        # EMPTY INPUT
        # ==========================================

        else:

            st.warning(
                "Please provide a URL or image."
            )

        # ==========================================
        # CONTINUE IF TEXT EXISTS
        # ==========================================

        if input_text != "":

            st.subheader("Extracted Text")

            st.write(
                input_text[:5000]
            )

            st.write(
                f"Extracted Character Count: {len(input_text)}"
            )

            # ======================================
            # PREPROCESSING
            # ======================================

            cleaned_text = clean_text(
                input_text
            )

            cleaned_text = remove_stopwords(
                cleaned_text
            )

            cleaned_text = lemmatize_text(
                cleaned_text
            )

            st.subheader(
                "Cleaned Text"
            )

            st.write(
                cleaned_text[:3000]
            )

            # ======================================
            # TF-IDF
            # ======================================

            vectorized_text = vectorizer.transform(
                [cleaned_text]
            )

            # ======================================
            # PREDICTION
            # ======================================

            prediction = model.predict(
                vectorized_text
            )[0]

            confidence = model.decision_function(
                vectorized_text
            )[0]

            score = abs(confidence)

            # ======================================
            # RESULT
            # ======================================

            st.subheader(
                "Prediction Result"
            )

            st.write(
                f"Confidence Score: {confidence:.2f}"
            )

            if score < 0.2:

                st.warning(
                    """
                    ⚠️ Inconclusive Analysis

                    The system could not confidently classify
                    this article as either real or fake news.

                    Additional verification is recommended.
                    """
                )

            elif prediction == 0:

                st.error(
                    "🚨 Fake News"
                )

            else:

                st.success(
                    "✅ Real News"
                )

    except Exception as e:

        st.error(
            "Could not analyze the provided news source."
        )

        st.write(e)