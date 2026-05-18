# 📰 Fake News Detection System

> AI-powered fake news detection web application built with Machine Learning, OCR, and NLP.

<br>

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![EasyOCR](https://img.shields.io/badge/EasyOCR-OCR-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 🚀 Live Demo

🔗 **Open App:**
https://fake-news-detector.streamlit.app

---

## 📌 About The Project

This project is an AI-powered fake news detection system that analyzes news articles and predicts whether they are **Fake** or **Real**.

The application supports:

* 🌐 News URL analysis
* 🖼️ Screenshot / image analysis using OCR
* 🤖 Machine learning predictions
* 🔐 Admin-controlled dataset management
* 🔄 Automatic model retraining

The system uses NLP preprocessing and a **Passive Aggressive Classifier** trained with TF-IDF vectorization.

---

## ✨ Features

✅ Analyze news directly from article links
✅ Upload screenshots of news articles
✅ OCR-based text extraction
✅ Fake / Real prediction system
✅ Confidence score display
✅ Automatic retraining after admin updates
✅ Admin panel for dataset management
✅ Duplicate news protection
✅ Edit or delete saved news
✅ Real-time model reload

---

## 🧠 Machine Learning Pipeline

### Text Preprocessing

* Lowercasing
* URL cleaning
* HTML removal
* Stopword removal
* Lemmatization

### Vectorization

* TF-IDF Vectorizer

### Models

* Logistic Regression
* Passive Aggressive Classifier

---

## 🛠️ Technologies Used

| Technology     | Purpose                   |
| -------------- | ------------------------- |
| Python         | Core programming language |
| Streamlit      | Web application           |
| Scikit-learn   | Machine learning          |
| Pandas & NumPy | Data processing           |
| NLTK           | NLP preprocessing         |
| EasyOCR        | OCR text extraction       |
| Newspaper3k    | News article extraction   |
| Joblib         | Model serialization       |

---

## 📂 Project Structure

```bash
fake-news-detector/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   └── (ignored)
│
├── models/
│   └── (ignored)
│
└── .gitignore
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/gulsunciftci/fake-news-detector.git
cd fake-news-detector
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🔐 Admin Panel

The admin system allows you to:

* Add new training data
* Upload images or URLs
* Assign Fake / Real labels
* Delete saved news
* Update labels
* Retrain the model automatically

---
## 📥 Dataset

The dataset files are not included in this repository because of GitHub file size limits.

Download the dataset from Kaggle:

🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

After downloading:

1. Extract the files
2. Place them inside the `data/` folder
---
## 👩‍💻 Author

### Gulsun Ciftci

🔗 GitHub:
https://github.com/gulsunciftci

---

## 📄 License

This project is licensed under the MIT License.
