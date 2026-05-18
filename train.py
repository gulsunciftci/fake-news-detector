import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score

# ==================================================
# LOAD DATASETS
# ==================================================

main_df = pd.read_csv(
    "data/final_cleaned_news.csv"
)

admin_df = pd.read_csv(
    "data/admin_data.csv"
)

# ==================================================
# MERGE DATASETS
# ==================================================

df = pd.concat(

    [main_df, admin_df],

    ignore_index=True

)

# ==================================================
# REMOVE DUPLICATES
# ==================================================

df.drop_duplicates(

    subset="clean_text",

    inplace=True

)

# ==================================================
# SAVE UPDATED DATASET
# ==================================================

df.to_csv(

    "data/updated_news_dataset.csv",

    index=False

)

# ==================================================
# FEATURES
# ==================================================

X = df["clean_text"]

y = df["label"]

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

# ==================================================
# TF-IDF
# ==================================================

vectorizer = TfidfVectorizer(

    max_df=0.7,

    min_df=2,

    ngram_range=(1,2),

    stop_words="english"

)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

# ==================================================
# LOGISTIC REGRESSION
# ==================================================

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_pred = lr_model.predict(
    X_test_tfidf
)

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)

# ==================================================
# PASSIVE AGGRESSIVE
# ==================================================

pa_model = PassiveAggressiveClassifier(
    max_iter=1000
)

pa_model.fit(
    X_train_tfidf,
    y_train
)

pa_pred = pa_model.predict(
    X_test_tfidf
)

pa_accuracy = accuracy_score(
    y_test,
    pa_pred
)

# ==================================================
# BEST MODEL
# ==================================================

if lr_accuracy > pa_accuracy:

    best_model = lr_model
    best_name = "Logistic Regression"
    best_accuracy = lr_accuracy

else:

    best_model = pa_model
    best_name = "PassiveAggressive"
    best_accuracy = pa_accuracy

# ==================================================
# RESULTS
# ==================================================

print("=" * 50)

print("BEST MODEL:", best_name)

print("ACCURACY:", best_accuracy)

print("=" * 50)

# ==================================================
# SAVE MODEL
# ==================================================

joblib.dump(

    best_model,

    "models/fake_news_model.pkl"

)

joblib.dump(

    vectorizer,

    "models/tfidf_vectorizer.pkl"

)

print("Model saved successfully.")