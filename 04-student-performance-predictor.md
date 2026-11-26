# Student Performance Predictor

**Level:** Beginner | **Estimated time:** 2 to 4 days | **Domain:** Tabular ML (regression/classification)

## What this project shows
Feature analysis on a relatable dataset, and awareness of data leakage (a key lesson for beginners).

## Dataset
UCI **Student Performance** dataset (Portuguese and math courses). Use `student-mat.csv`, which is semicolon-separated. Target is `G3` (final grade).

## Tech stack
Python, pandas, scikit-learn, seaborn, Streamlit (optional).

## Suggested folder structure
```
student-performance/
├── data/
│   └── student-mat.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   └── train.py
├── requirements.txt
└── README.md
```

## Step-by-step plan
1. Load with `sep=";"` and inspect columns (study time, failures, absences, parents' education, and so on).
2. EDA: how study time, absences, and past failures relate to final grade.
3. Notice that `G1` and `G2` (earlier grades) are extremely predictive. Train once with them and once without, and explain the difference.
4. One-hot encode categoricals and train Linear Regression and Random Forest.
5. Optionally convert `G3` to pass/fail (>= 10) and train a classifier.
6. Report which non-grade features matter most and what that suggests for early-warning systems.

## Starter code (`src/train.py`)
Treat this as a working skeleton. Run it, understand each line, then improve it.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/student-mat.csv", sep=";")

def run(drop_cols, label):
    y = df["G3"]
    X = pd.get_dummies(df.drop(columns=drop_cols), drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"{label}: MAE={mean_absolute_error(y_test, preds):.2f}  R2={r2_score(y_test, preds):.2f}")

run(["G3"], "With G1 and G2")
run(["G3", "G1", "G2"], "Without G1 and G2")
```

## What to put in your README
- **Problem statement:** one paragraph on what you are predicting or building, and why it matters.
- **Dataset:** name, source link, size, and any cleaning you did.
- **Approach:** models tried and why.
- **Results:**
- Metrics with and without `G1`/`G2`, and your explanation of the gap.
- Feature importance chart for the no-previous-grades model.
- A short note on ethical use (this should support students, not label them).
- **How to run:** `pip install -r requirements.txt`, then the exact command(s).
- **Limitations and next steps:** be honest about what the model gets wrong.

## Ways to extend it (after the basic version works)
- Combine math and Portuguese datasets.
- Add a Streamlit app for what-if scenarios (for example, more study time).
- Try classification (pass/fail) and compare metrics.

## Resume line
Analyzed student performance data and built regression models in scikit-learn, demonstrating the impact of data leakage and identifying key non-grade predictors of outcomes.

## Before you push to GitHub
- Add a `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, large model files).
- Do not commit large datasets; link to them and add a `download_data` note in the README.
- Never commit API keys. Use environment variables.
- Commit in small steps with clear messages (for example "Add baseline model", "Add evaluation plots").
