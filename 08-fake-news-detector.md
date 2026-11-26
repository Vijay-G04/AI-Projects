# Fake News / Toxic Comment Detector

**Level:** Beginner | **Estimated time:** 3 to 5 days | **Domain:** NLP (text classification)

## What this project shows
Text classification on real-world text, and critical thinking about dataset bias and leakage.

## Dataset
- **Fake news:** "Fake and real news dataset" (ISOT) on Kaggle, with `Fake.csv` and `True.csv`.
- **Toxic comments (alternative):** Jigsaw Toxic Comment Classification Challenge on Kaggle (`comment_text` plus toxicity labels).

## Tech stack
Python, pandas, scikit-learn, Gradio.

## Suggested folder structure
```
fake-news-detector/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   └── train.py
├── models/
├── app.py
├── requirements.txt
└── README.md
```

## Step-by-step plan
1. Load both CSVs, add a label column (0 = real, 1 = fake), and combine them.
2. Check for leakage. In the ISOT data, real articles often contain publisher tags such as a news agency name at the start. A model can learn those shortcuts instead of real signals.
3. Remove such artifacts and compare accuracy before and after. This is a great insight to write up.
4. Train TF-IDF with Logistic Regression and a linear SVM.
5. Evaluate with precision, recall, F1, and look at the most influential words.
6. Test on text from outside the dataset and report honestly how well it generalizes.
7. Build a Gradio app that takes an article and returns a prediction with a clear disclaimer.

## Starter code (`src/train.py`)
Treat this as a working skeleton. Run it, understand each line, then improve it.

```python
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")
fake["label"], real["label"] = 1, 0

df = pd.concat([fake, real], ignore_index=True)
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

# IMPORTANT: inspect and remove publisher tags / leaked patterns here.

X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

model = make_pipeline(
    TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2), min_df=5),
    LogisticRegression(max_iter=1000),
)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
joblib.dump(model, "models/fake_news_model.joblib")
```

## What to put in your README
- **Problem statement:** one paragraph on what you are predicting or building, and why it matters.
- **Dataset:** name, source link, size, and any cleaning you did.
- **Approach:** models tried and why.
- **Results:**
- Metrics before and after removing leakage.
- Top words for each class.
- Results on a few external articles, with honest commentary.
- **How to run:** `pip install -r requirements.txt`, then the exact command(s).
- **Limitations and next steps:** be honest about what the model gets wrong.

## Ways to extend it (after the basic version works)
- Fine-tune a transformer and compare.
- Add explanations with LIME.
- Switch to the Jigsaw toxic comment dataset (multi-label).

## Resume line
Built a fake news classifier with TF-IDF and logistic regression, diagnosing dataset leakage to produce a more honest evaluation; served via Gradio.

## Before you push to GitHub
- Add a `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, large model files).
- Do not commit large datasets; link to them and add a `download_data` note in the README.
- Never commit API keys. Use environment variables.
- Commit in small steps with clear messages (for example "Add baseline model", "Add evaluation plots").
