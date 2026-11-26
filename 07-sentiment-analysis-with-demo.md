# Sentiment Analysis (Baseline vs Transformer) with Streamlit Demo

**Level:** Beginner | **Estimated time:** 4 to 6 days | **Domain:** NLP

## What this project shows
Building a classical baseline, using a pretrained transformer from Hugging Face, comparing the two honestly, and shipping a demo.

## Dataset
**IMDB movie reviews** (50,000 labeled reviews). Load with Hugging Face `datasets` (`load_dataset("imdb")`) or from Kaggle.

## Tech stack
Python, scikit-learn, Hugging Face `transformers` and `datasets`, Streamlit.

## Suggested folder structure
```
sentiment-analysis/
├── notebooks/
│   └── baseline_vs_transformer.ipynb
├── src/
│   ├── baseline.py
│   └── transformer_eval.py
├── app.py               # Streamlit demo
├── requirements.txt
└── README.md
```

## Step-by-step plan
1. Load IMDB and inspect a few reviews and the label balance.
2. Build a baseline: TF-IDF with Logistic Regression. Record accuracy and F1.
3. Run a pretrained sentiment pipeline from Hugging Face on a sample of the test set (2,000 reviews is enough on a laptop).
4. Compare accuracy, speed, and model size between the baseline and the transformer.
5. Inspect failure cases: sarcasm, negation, mixed reviews.
6. Build a Streamlit app where a user types a review and sees the label and confidence.
7. Document the trade-offs: when is the simple model good enough?

## Starter code (`src/baseline.py`)
Treat this as a working skeleton. Run it, understand each line, then improve it.

```python
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

ds = load_dataset("imdb")
X_train, y_train = ds["train"]["text"], ds["train"]["label"]
X_test, y_test = ds["test"]["text"], ds["test"]["label"]

model = make_pipeline(
    TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True),
    LogisticRegression(max_iter=1000),
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, preds), 4))
print("F1:", round(f1_score(y_test, preds), 4))


# ------------------------------------------------------------
# app.py (Streamlit demo)
# ------------------------------------------------------------
# import streamlit as st
# from transformers import pipeline
#
# @st.cache_resource
# def load_model():
#     return pipeline("sentiment-analysis",
#                     model="distilbert-base-uncased-finetuned-sst-2-english")
#
# st.title("Sentiment Analyzer")
# text = st.text_area("Enter a review")
# if st.button("Analyze") and text.strip():
#     result = load_model()(text[:1000], truncation=True)[0]
#     st.write(f"**{result['label']}** (confidence {result['score']:.2f})")
```

## What to put in your README
- **Problem statement:** one paragraph on what you are predicting or building, and why it matters.
- **Dataset:** name, source link, size, and any cleaning you did.
- **Approach:** models tried and why.
- **Results:**
- A table comparing baseline and transformer on accuracy, F1, and inference time.
- Five failure examples with your explanation.
- A screenshot of the demo app.
- **How to run:** `pip install -r requirements.txt`, then the exact command(s).
- **Limitations and next steps:** be honest about what the model gets wrong.

## Ways to extend it (after the basic version works)
- Fine-tune DistilBERT on IMDB yourself.
- Try a three-class dataset (positive, neutral, negative).
- Analyze reviews from a product category you care about.
- Deploy on Hugging Face Spaces.

## Resume line
Compared TF-IDF + logistic regression against a pretrained DistilBERT for sentiment analysis on IMDB and deployed an interactive Streamlit app.

## Before you push to GitHub
- Add a `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, large model files).
- Do not commit large datasets; link to them and add a `download_data` note in the README.
- Never commit API keys. Use environment variables.
- Commit in small steps with clear messages (for example "Add baseline model", "Add evaluation plots").
