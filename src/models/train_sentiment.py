from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "reviews_clean.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "nlp"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df["review"].astype(str)
y = df["sentiment"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print(classification_report(y_test, preds))

joblib.dump(model, ARTIFACT_DIR / "sentiment_model.pkl")
joblib.dump(vectorizer, ARTIFACT_DIR / "tfidf_vectorizer.pkl")
print("[OK] Sentiment model saved")
