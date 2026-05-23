import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score


# LOAD DATASETS


main_df = pd.read_csv(  # Ana temizlenmiş dataset yükleniyor. Bu büyük eğitim datası.
    "data/final_cleaned_news.csv"
)

admin_df = pd.read_csv( # Admin panelinden eklenen haberler yükleniyor.
    "data/admin_data.csv"
)

# MERGE DATASETS


df = pd.concat(

    [main_df, admin_df], # İki dataset birleşiyor.

    ignore_index=True

)

# REMOVE DUPLICATES


df.drop_duplicates( # Aynı haberleri kaldırır.

    subset="clean_text",  # Temizlenmiş metne göre tekrar eden haberleri bulur.

    inplace=True # Değişikliği direkt dataframe üzerinde yapar.

)


# SAVE UPDATED DATASET


df.to_csv(  # Dataseti CSV olarak kaydeder.

    "data/updated_news_dataset.csv",

    index=False

)


# FEATURES


X = df["clean_text"] # Modelin giriş verileri. Yani haber metinleri.

y = df["label"] # Tahmin edilmesi gereken değerler. 0 → Fake 1 → Real


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(  # Dataseti eğitim ve test olarak ayırır.

    X,  # Metinler ve label’lar.
    y,

    test_size=0.2,  # Verinin: %80 eğitim %20 test olarak ayrılmasını sağlar.

    random_state=42, # Her çalıştırmada aynı sonucu verir.

    stratify=y # Fake ve real oranını dengeli tutar.

)


# TF-IDF


vectorizer = TfidfVectorizer(  # Metni sayısal hale çevirir.

    max_df=0.7, # Çok sık geçen kelimeleri ignore eder.

    min_df=2, # Çok nadir geçen kelimeleri ignore eder.

    ngram_range=(1,2),

    stop_words="english"

)

X_train_tfidf = vectorizer.fit_transform(  # TF-IDF: train datasını öğrenir sayısal vektöre dönüştürür
    X_train
)

X_test_tfidf = vectorizer.transform( # Test datasını aynı sisteme göre dönüştürür.
    X_test
)


# LOGISTIC REGRESSION


lr_model = LogisticRegression( # Logistic Regression modeli oluşturur.
    max_iter=1000 # Modelin maksimum öğrenme sayısı.
)

lr_model.fit( # Model eğitilir.
    X_train_tfidf,
    y_train
)

lr_pred = lr_model.predict(  # Test verileri üzerinde tahmin yapar.
    X_test_tfidf
)

lr_accuracy = accuracy_score( # Doğruluk oranını hesaplar.
    y_test,
    lr_pred
)


# PASSIVE AGGRESSIVE


pa_model = PassiveAggressiveClassifier( # İkinci modeli oluşturur.
    max_iter=1000  
)

pa_model.fit(  # Model eğitilir.
    X_train_tfidf,
    y_train
)

pa_pred = pa_model.predict( # Test datası üzerinde tahmin yapar.
    X_test_tfidf
)

pa_accuracy = accuracy_score(  # Başarı oranını hesaplar.
    y_test,
    pa_pred
)


# BEST MODEL


if lr_accuracy > pa_accuracy:  # Hangi model daha başarılı kontrol edilir.

    best_model = lr_model
    best_name = "Logistic Regression"  # En iyi model Logistic Regression olur.
    best_accuracy = lr_accuracy

else:

    best_model = pa_model
    best_name = "PassiveAggressive" # En iyi model PassiveAggressive olur.
    best_accuracy = pa_accuracy


# RESULTS


print("=" * 50)  # Ekrana ayırıcı çizgi basar.

print("BEST MODEL:", best_name) # Kazanan modeli gösterir.

print("ACCURACY:", best_accuracy) # Başarı oranını yazdırır.

print("=" * 50)

# ==================================================
# SAVE MODEL
# ==================================================

joblib.dump(  # Modeli .pkl dosyası olarak kaydeder.

    best_model,

    "models/fake_news_model.pkl"

)

joblib.dump( # TF-IDF vectorizer da kaydedilir.

    vectorizer,

    "models/tfidf_vectorizer.pkl"

)

print("Model saved successfully.")