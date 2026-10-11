"""
train_models.py
-----------------
Trains every model used by the AI IT Helpdesk project and saves them into
models/ with joblib, so app.py never has to retrain anything.

Run this ONCE after you have the dataset in data/, and again any time the
dataset changes:

    python train_models.py

What it trains and saves:
  1. CATEGORY classifier -- TF-IDF + Linear SVM (primary model), and a
     Logistic Regression model trained on the exact same train/test
     split, so the two can be fairly compared.
  2. SUBCATEGORY classifiers -- one TF-IDF + Linear SVM per category that
     has enough data (SOFTWARE, PIV CARD, PRINTER, NETWORK, ACCOUNT,
     SECURITY). EMAIL and HARDWARE use a documented rule instead -- see
     helpdesk_core.RULE_BASED_SUBCATEGORY for why.
  3. ROUTING lookup table (category, subcategory) -> assignment group.
  4. SOLUTION RECOMMENDATION search index (TF-IDF + cosine similarity
     over historical resolved tickets).

All evaluation numbers printed here, and saved to
models/evaluation_results.json, come from a genuine held-out test split
(20% of the data, never seen during training). The TF-IDF vectorizer is
fit ONLY on the training rows in every case, so nothing about the test
set leaks into training.
"""

import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from helpdesk_core import clean_text, build_input_text, save_artifact

DATA_PATH = "data/synthetic-it-call-center-tickets.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# For each category: which text field to use for its subcategory model,
# and which subcategories have enough rows to be worth training on.
# (SOFTWARE and SECURITY drop a few subcategories with only a handful of
# rows each -- too few for a stratified train/test split to be meaningful.)
SUBCATEGORY_CONFIG = {
    "SOFTWARE": {
        "text_field": "model_text",
        "keep_subcategories": ["ERROR", "INSTALLATION", "MALFUNCTION", "CONFIGURATION", "ACCESS", "TRAINING"],
    },
    "PIV CARD": {"text_field": "short_description", "keep_subcategories": None},
    "PRINTER":  {"text_field": "short_description", "keep_subcategories": None},
    "NETWORK":  {"text_field": "short_description", "keep_subcategories": None},
    "ACCOUNT":  {"text_field": "short_description", "keep_subcategories": None},
    "SECURITY": {"text_field": "short_description", "keep_subcategories": ["ACCESS", "ERROR", "CONFIGURATION"]},
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna(subset=["category", "subcategory"])

    df["clean_text"] = (df["short_description"].astype(str) + " " + df["content"].astype(str)).apply(clean_text)
    df["model_text"] = df.apply(
        lambda r: build_input_text(r["short_description"], r["type"], r["issue/request"], r["software/system"]),
        axis=1,
    )
    return df


def evaluate(name, y_true, y_pred, results_log):
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n--- {name} ---")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro):    {rec_macro:.4f}")
    print(f"F1 (macro):        {f1_macro:.4f}")
    print(f"F1 (weighted):     {f1_weighted:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))

    results_log[name] = {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }
    return acc, f1_macro


def train_category_model(df, results_log):
    print("\n========== TRAINING: TICKET CATEGORY (SVM vs Logistic Regression) ==========")
    X, y = df["model_text"], df["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)   # fit ONLY on train
    X_test_tfidf = vectorizer.transform(X_test)

    # --- Primary model: Linear SVM ---
    svm_model = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    svm_model.fit(X_train_tfidf, y_train)
    svm_pred = svm_model.predict(X_test_tfidf)
    evaluate("CATEGORY - Linear SVM", y_test, svm_pred, results_log)

    cm = confusion_matrix(y_test, svm_pred, labels=svm_model.classes_)
    print("Confusion matrix (SVM) -- rows = actual, columns = predicted:")
    print(pd.DataFrame(cm, index=svm_model.classes_, columns=svm_model.classes_))

    # --- Logistic Regression comparison (a classical model, same family you've studied) ---
    logreg_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    logreg_model.fit(X_train_tfidf, y_train)
    logreg_pred = logreg_model.predict(X_test_tfidf)
    evaluate("CATEGORY - Logistic Regression", y_test, logreg_pred, results_log)

    # Save the production artifacts (SVM is the one app.py actually uses)
    save_artifact(vectorizer, "category_vectorizer.joblib")
    save_artifact(svm_model, "category_model.joblib")
    save_artifact(logreg_model, "category_model_logreg.joblib")  # kept only for the report/comparison


def train_subcategory_models(df, results_log):
    print("\n========== TRAINING: SUBCATEGORY MODELS (per category) ==========")
    subcategory_models = {}

    for category, cfg in SUBCATEGORY_CONFIG.items():
        cat_df = df[df["category"] == category].copy()

        if cfg["keep_subcategories"]:
            cat_df = cat_df[cat_df["subcategory"].isin(cfg["keep_subcategories"])]

        counts = cat_df["subcategory"].value_counts()
        valid_classes = counts[counts >= 2].index  # stratify needs >=2 rows per class
        cat_df = cat_df[cat_df["subcategory"].isin(valid_classes)]

        if cat_df["subcategory"].nunique() < 2 or len(cat_df) < 10:
            print(f"\nSkipping subcategory model for {category}: not enough data "
                  f"({len(cat_df)} rows, {cat_df['subcategory'].nunique()} usable classes).")
            continue

        text_col = cfg["text_field"]
        X = cat_df[text_col].astype(str)
        y = cat_df["subcategory"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        vec = TfidfVectorizer(max_features=10000, stop_words="english")
        X_train_tfidf = vec.fit_transform(X_train)
        X_test_tfidf = vec.transform(X_test)

        model = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
        model.fit(X_train_tfidf, y_train)
        pred = model.predict(X_test_tfidf)

        evaluate(f"SUBCATEGORY - {category}", y_test, pred, results_log)

        subcategory_models[category] = {"vectorizer": vec, "model": model, "text_field": text_col}

    save_artifact(subcategory_models, "subcategory_models.joblib")


def build_routing_lookup(df):
    print("\n========== BUILDING: ROUTING LOOKUP TABLE ==========")
    routing_summary = (
        df.dropna(subset=["assignment_group"])
          .groupby(["category", "subcategory", "assignment_group"])
          .size()
          .reset_index(name="count")
    )
    routing_summary["total"] = routing_summary.groupby(["category", "subcategory"])["count"].transform("sum")
    routing_summary["percentage"] = routing_summary["count"] / routing_summary["total"] * 100

    dominant = (
        routing_summary.sort_values(["category", "subcategory", "count"], ascending=[True, True, False])
        .groupby(["category", "subcategory"]).first().reset_index()
    )

    routing_lookup = {}
    for _, row in dominant.iterrows():
        confidence = row["percentage"]
        if confidence >= 70:
            status = "HIGH CONFIDENCE"
        elif confidence >= 50:
            status = "MEDIUM CONFIDENCE"
        else:
            status = "MANUAL REVIEW"
        routing_lookup[(row["category"], row["subcategory"])] = {
            "assignment_group": row["assignment_group"],
            "confidence": confidence,
            "status": status,
        }

    print(f"Routing rules built for {len(routing_lookup)} (category, subcategory) combinations.")
    save_artifact(routing_lookup, "routing_lookup.joblib")


def build_recommendation_index(df):
    print("\n========== BUILDING: SOLUTION RECOMMENDATION INDEX ==========")
    solution_df = df.dropna(subset=["close_notes"]).copy()
    solution_df["search_text"] = (
        solution_df["short_description"].astype(str) + " " +
        solution_df["type"].astype(str) + " " +
        solution_df["issue/request"].astype(str) + " " +
        solution_df["software/system"].astype(str)
    ).apply(clean_text)

    search_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    search_vectorizer.fit(solution_df["search_text"])

    print(f"Recommendation index built from {len(solution_df)} resolved historical tickets.")

    save_artifact(search_vectorizer, "search_vectorizer.joblib")
    save_artifact(
        solution_df[["number", "short_description", "category", "subcategory", "close_notes", "search_text"]],
        "solution_df.joblib",
    )


def main():
    print("Loading dataset from", DATA_PATH)
    df = load_data()
    print(f"Loaded {len(df)} tickets after dropping rows with missing category/subcategory.")

    results_log = {}

    train_category_model(df, results_log)
    train_subcategory_models(df, results_log)
    build_routing_lookup(df)
    build_recommendation_index(df)

    with open("models/evaluation_results.json", "w") as f:
        json.dump(results_log, f, indent=2)

    print("\n================================================")
    print("All models trained and saved to the 'models/' folder.")
    print("Evaluation metrics written to models/evaluation_results.json")
    print("Run 'python app.py' to launch the Gradio demo.")
    print("================================================")


if __name__ == "__main__":
    main()
