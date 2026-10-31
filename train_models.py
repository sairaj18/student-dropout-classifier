"""
Trains 5 classifiers, evaluates them, and saves everything the app needs.
Run once:  python train_models.py
Dataset: UCI "Predict Students' Dropout and Academic Success"
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from metrics_utils import compute_metrics

TARGET = "Target"
SEED = 42
os.makedirs("model", exist_ok=True)


def load_data():
    """Try UCI download first; fall back to a local CSV named students_raw.csv."""
    if os.path.exists("students_raw.csv"):
        return pd.read_csv("students_raw.csv")
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=697)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv("students_raw.csv", index=False)
    return df


df = load_data()
df.columns = [c.strip() for c in df.columns]
if TARGET not in df.columns:            # safety: some copies name it differently
    df = df.rename(columns={df.columns[-1]: TARGET})
print("Dataset shape:", df.shape)
print(df[TARGET].value_counts())

X = df.drop(columns=[TARGET])
y_text = df[TARGET]
le = LabelEncoder()
y = le.fit_transform(y_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# Save the test set (original text labels) for upload into the Streamlit app
test_df = X_test.copy()
test_df[TARGET] = le.inverse_transform(y_test)
test_df.to_csv("test_data.csv", index=False)

models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
    "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(max_depth=8, random_state=SEED))]),
    "KNN": Pipeline([("scale", StandardScaler()),
                     ("clf", KNeighborsClassifier(n_neighbors=7))]),
    "Naive Bayes": Pipeline([("clf", GaussianNB())]),
    "Support Vector Machine": Pipeline([("scale", StandardScaler()),
                                        ("clf", SVC(kernel="rbf", probability=True, random_state=SEED))]),
}

rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    m = compute_metrics(y_test, pred, proba)
    rows.append({"ML Model Name": name, **m})
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join("model", fname))
    print(f"{name:25s}", {k: round(v, 4) for k, v in m.items()})

joblib.dump(le, "model/label_encoder.joblib")
joblib.dump(list(X.columns), "model/feature_columns.joblib")

results = pd.DataFrame(rows).round(4)
results.to_csv("model/results_table.csv", index=False)
try:
    print(results.to_markdown(index=False))
except ImportError:          # tabulate not installed
    print(results.to_string(index=False))
