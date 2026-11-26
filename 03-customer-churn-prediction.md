# Customer Churn Prediction

**Level:** Beginner | **Estimated time:** 4 to 6 days | **Domain:** Tabular ML (classification)

## What this project shows
Handling categorical features, imbalanced classes, choosing the right metric, and explaining model results in business terms.

## Dataset
**Telco Customer Churn** (IBM sample dataset, available on Kaggle). About 7,000 customers with contract, billing, and service columns, and a `Churn` Yes/No target.

## Tech stack
Python, pandas, scikit-learn, matplotlib/seaborn, optionally XGBoost and SHAP.

## Suggested folder structure
```
churn-prediction/
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── models/
├── requirements.txt
└── README.md
```

## Step-by-step plan
1. Load data. Note that `TotalCharges` is stored as text and has blank values you must handle.
2. EDA: churn rate by contract type, tenure, payment method, and monthly charges.
3. Encode categorical columns with one-hot encoding and scale numeric ones where needed.
4. Split with stratification so train and test keep the same churn ratio.
5. Train Logistic Regression (with `class_weight="balanced"`), Random Forest, and Gradient Boosting.
6. Evaluate with ROC-AUC, precision, recall, and a precision-recall curve. Explain why accuracy is not enough.
7. Interpret the model: top coefficients or feature importances, and write 3 plain-language business insights.

## Starter code (`src/train.py`)
Treat this as a working skeleton. Run it, understand each line, then improve it.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna().drop(columns=["customerID"])

y = (df["Churn"] == "Yes").astype(int)
X = pd.get_dummies(df.drop(columns=["Churn"]), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)
print("ROC-AUC:", round(roc_auc_score(y_test, probs), 3))
print(classification_report(y_test, preds))

coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
print(coefs.head(5))
print(coefs.tail(5))
```

## What to put in your README
- **Problem statement:** one paragraph on what you are predicting or building, and why it matters.
- **Dataset:** name, source link, size, and any cleaning you did.
- **Approach:** models tried and why.
- **Results:**
- ROC-AUC and recall for the churn class across models.
- A bar chart of the top factors driving churn.
- Three business recommendations based on the findings.
- **How to run:** `pip install -r requirements.txt`, then the exact command(s).
- **Limitations and next steps:** be honest about what the model gets wrong.

## Ways to extend it (after the basic version works)
- Try SMOTE or class weighting and compare.
- Tune the decision threshold for a chosen recall target.
- Use SHAP for per-customer explanations.
- Build a Streamlit page that scores a single customer.

## Resume line
Built a customer churn model on telecom data (ROC-AUC X) handling class imbalance and categorical features; identified key churn drivers to support retention recommendations.

## Before you push to GitHub
- Add a `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, large model files).
- Do not commit large datasets; link to them and add a `download_data` note in the README.
- Never commit API keys. Use environment variables.
- Commit in small steps with clear messages (for example "Add baseline model", "Add evaluation plots").
