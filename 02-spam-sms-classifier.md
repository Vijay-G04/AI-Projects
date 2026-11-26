# Spam / SMS Classifier

**Level:** Beginner | **Estimated time:** 2 to 4 days | **Domain:** NLP (text classification)

## What this project shows
Text preprocessing, TF-IDF features, Naive Bayes and logistic regression, and evaluating classifiers where the classes are imbalanced.

## Dataset
UCI **SMS Spam Collection** (about 5,600 messages labeled ham or spam). It is also mirrored on Kaggle. The file is tab-separated: `label<TAB>text`.

## Tech stack
Python, pandas, scikit-learn, NLTK (optional), matplotlib, Gradio or Streamlit.

## Suggested folder structure
```
spam-classifier/
├── data/
│   └── SMSSpamCollection
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── models/
├── app.py
├── requirements.txt
└── README.md
```

## Step-by-step plan
1. Load the data and check class balance (spam is a small minority).
2. Explore: message length by class, most common words in spam vs ham, word clouds.
3. Clean text lightly (lowercase, optional stopword removal). Avoid over-cleaning, since punctuation and digits can signal spam.
4. Build a pipeline: `TfidfVectorizer` followed by `MultinomialNB`.
5. Compare with Logistic Regression and a linear SVM.
6. Evaluate with precision, recall, F1, and a confusion matrix. Explain why accuracy alone is misleading here.
7. Save the pipeline and build a small app where a user pastes a message and gets spam probability.

## Starter code (`src/train.py`)
Treat this as a working skeleton. Run it, understand each line, then improve it.

```python
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "text"])
print(df["label"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

joblib.dump(model, "models/spam_model.joblib")
```

## What to put in your README
- **Problem statement:** one paragraph on what you are predicting or building, and why it matters.
- **Dataset:** name, source link, size, and any cleaning you did.
- **Approach:** models tried and why.
- **Results:**
- Precision, recall, and F1 for the spam class.
- A confusion matrix.
- A short list of the words most associated with spam.
- **How to run:** `pip install -r requirements.txt`, then the exact command(s).
- **Limitations and next steps:** be honest about what the model gets wrong.

## Ways to extend it (after the basic version works)
- Add character n-grams to the TF-IDF features.
- Fine-tune a small transformer (DistilBERT) and compare against Naive Bayes.
- Handle threshold tuning to trade precision against recall.
- Wrap it in a Gradio app with example messages.

## Resume line
Developed an SMS spam classifier using TF-IDF and Naive Bayes reaching X% F1 on the spam class; served predictions through a Gradio web app.

## Before you push to GitHub
- Add a `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, large model files).
- Do not commit large datasets; link to them and add a `download_data` note in the README.
- Never commit API keys. Use environment variables.
- Commit in small steps with clear messages (for example "Add baseline model", "Add evaluation plots").
