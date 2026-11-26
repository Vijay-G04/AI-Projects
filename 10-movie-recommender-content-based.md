# Movie Recommender (Content-Based)

**Level:** Beginner | **Estimated time:** 2 to 4 days | **Domain:** Recommender systems / NLP

## What this project shows
Text vectorization, cosine similarity, and how content-based recommendation works.

## Dataset
**TMDB 5000 Movie Dataset** on Kaggle (`tmdb_5000_movies.csv`, optionally `tmdb_5000_credits.csv` for cast and crew).

## Tech stack
Python, pandas, scikit-learn, Streamlit.

## Suggested folder structure
```
movie-recommender/
├── data/
│   └── tmdb_5000_movies.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   └── recommender.py
├── app.py
├── requirements.txt
└── README.md
```

## Step-by-step plan
1. Load the movies file and inspect the `overview` and `genres` columns (genres are stored as JSON-like strings).
2. Parse genres into plain text and combine them with the overview into one text field.
3. Convert text to vectors with TF-IDF.
4. Compute cosine similarity between all movies.
5. Write a `recommend(title, n)` function that returns the n most similar movies.
6. Sanity-check the results with movies you know and note where it fails (for example, it ignores popularity and quality).
7. Build a Streamlit app with a movie dropdown and recommendations list.

## Starter code (`src/recommender.py`)
Treat this as a working skeleton. Run it, understand each line, then improve it.

```python
import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/tmdb_5000_movies.csv")
df["overview"] = df["overview"].fillna("")
df["genres_text"] = df["genres"].apply(
    lambda s: " ".join(g["name"] for g in ast.literal_eval(s))
)
df["soup"] = df["overview"] + " " + df["genres_text"]

tfidf = TfidfVectorizer(stop_words="english")
matrix = tfidf.fit_transform(df["soup"])
similarity = cosine_similarity(matrix)

indices = pd.Series(df.index, index=df["title"]).drop_duplicates()


def recommend(title, n=5):
    i = indices[title]
    scores = sorted(enumerate(similarity[i]), key=lambda x: x[1], reverse=True)[1 : n + 1]
    return df["title"].iloc[[j for j, _ in scores]].tolist()


if __name__ == "__main__":
    print(recommend("The Dark Knight Rises"))
```

## What to put in your README
- **Problem statement:** one paragraph on what you are predicting or building, and why it matters.
- **Dataset:** name, source link, size, and any cleaning you did.
- **Approach:** models tried and why.
- **Results:**
- Example recommendations for 3 or 4 well-known movies.
- A short discussion of where content-based filtering works and fails.
- A screenshot of the app.
- **How to run:** `pip install -r requirements.txt`, then the exact command(s).
- **Limitations and next steps:** be honest about what the model gets wrong.

## Ways to extend it (after the basic version works)
- Add cast, director, and keywords to the text features.
- Blend similarity with popularity or rating.
- Try sentence embeddings instead of TF-IDF.
- Build a collaborative filtering version on MovieLens and compare.

## Resume line
Built a content-based movie recommender using TF-IDF and cosine similarity on 5,000 films, with an interactive Streamlit interface.

## Before you push to GitHub
- Add a `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, large model files).
- Do not commit large datasets; link to them and add a `download_data` note in the README.
- Never commit API keys. Use environment variables.
- Commit in small steps with clear messages (for example "Add baseline model", "Add evaluation plots").
