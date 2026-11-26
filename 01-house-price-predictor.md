# House Price Predictor

**Level:** Beginner | **Estimated time:** 3 to 5 days | **Domain:** Tabular ML (regression)

## What this project shows
Data cleaning, feature engineering, training several regression models, and comparing them with proper metrics.

## Dataset
- **Easiest:** California Housing, built into scikit-learn (`fetch_california_housing`, downloads on first use).
- **More challenging:** Kaggle "House Prices: Advanced Regression Techniques" (Ames housing), which has many missing values and categorical columns.

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib/seaborn, Streamlit (optional demo).

## Suggested folder structure
```
house-price-predictor/
├── data/                # raw data (gitignored)
├── notebooks/
│   └── eda.ipynb        # exploration and plots
├── src/
│   ├── train.py         # training and evaluation
│   └── predict.py       # load model, predict one example
├── models/              # saved model (joblib)
├── app.py               # optional Streamlit demo
├── requirements.txt
└── README.md
```

## Step-by-step plan
1. Load the data and inspect shape, dtypes, missing values, and the target distribution.
2. Do EDA: histograms, correlation heatmap, scatter plots of the strongest features against price.
3. Split into train and test sets (80/20) before any fitting to avoid leakage.
4. Train a baseline Linear Regression, then Random Forest, then Gradient Boosting.
5. Evaluate with MAE, RMSE, and R². Plot predicted vs actual values and residuals.
6. Look at feature importances and explain what drives price.
7. Save the best model with joblib and write a `predict.py` that takes inputs and returns a price.
8. Optional: build a Streamlit form where a user enters features and sees a predicted price.

## Starter code (`src/train.py`)
Treat this as a working skeleton. Run it, understand each line, then improve it.

```python
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}

best_name, best_score = None, -1
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE={mae:.3f}  R2={r2:.3f}")
    if r2 > best_score:
        best_name, best_score = name, r2

joblib.dump(models[best_name], "models/best_model.joblib")
print("Saved:", best_name)
```

## What to put in your README
- **Problem statement:** one paragraph on what you are predicting or building, and why it matters.
- **Dataset:** name, source link, size, and any cleaning you did.
- **Approach:** models tried and why.
- **Results:**
- A table comparing each model's MAE, RMSE, and R².
- A predicted-vs-actual scatter plot.
- A feature importance bar chart.
- **How to run:** `pip install -r requirements.txt`, then the exact command(s).
- **Limitations and next steps:** be honest about what the model gets wrong.

## Ways to extend it (after the basic version works)
- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`.
- Try XGBoost or LightGBM and compare.
- Add cross-validation instead of a single split.
- Explain predictions with SHAP values.
- Deploy the Streamlit app on Streamlit Community Cloud.

## Resume line
Built and compared regression models (Linear Regression, Random Forest) on housing data using scikit-learn, achieving R² of X; deployed an interactive price predictor with Streamlit.

## Before you push to GitHub
- Add a `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, large model files).
- Do not commit large datasets; link to them and add a `download_data` note in the README.
- Never commit API keys. Use environment variables.
- Commit in small steps with clear messages (for example "Add baseline model", "Add evaluation plots").
