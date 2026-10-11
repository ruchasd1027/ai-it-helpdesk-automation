"""
helpdesk_core.py
-----------------
Shared logic for the AI IT Helpdesk project.

Both train_models.py (training) and app.py (the Gradio demo) import from
this one file, so there is only ONE copy of the text-cleaning rules and
ONE copy of the prediction logic. This is what fixes the original bug
where the Gradio app referred to models (printer_model, network_model,
account_model, ...) that were only defined later in the notebook.
"""

import re
import os
import joblib

MODELS_DIR = "models"

# ---------------------------------------------------------------------
# Text cleaning (identical to the original notebook)
# ---------------------------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_input_text(short_description, ticket_type="", issue_request="", software_system=""):
    text = (
        str(short_description) + " " +
        str(ticket_type) + " " +
        str(issue_request) + " " +
        str(software_system)
    )
    return clean_text(text)


# ---------------------------------------------------------------------
# Categories where a trained subcategory model is NOT used.
# This is a deliberate, documented decision, not a missing feature:
#   - EMAIL: 380 of 382 historical EMAIL tickets (99.5%) are "REQUEST".
#            A trained model would just learn to always say REQUEST anyway.
#   - HARDWARE: only 38 historical tickets total, split across 5
#            subcategories. Too little data to train anything reliable,
#            so these go to a human for manual review instead of guessing.
# ---------------------------------------------------------------------
RULE_BASED_SUBCATEGORY = {
    "EMAIL": "REQUEST",
    "HARDWARE": "MANUAL REVIEW",
}


def artifact_path(name):
    return os.path.join(MODELS_DIR, name)


def save_artifact(obj, name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(obj, artifact_path(name))
    print(f"Saved: {artifact_path(name)}")


def load_artifact(name):
    path = artifact_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing model file: '{path}'. "
            f"Run 'python train_models.py' first to train and save all models."
        )
    return joblib.load(path)


class HelpdeskModels:
    """
    Loads every trained artifact ONCE and exposes predict_ticket(), exactly
    like the original notebook's predict_ticket() function -- but backed by
    files on disk instead of in-memory notebook variables. This is what
    lets the Gradio app start instantly without retraining.
    """

    def __init__(self):
        self.category_vectorizer = load_artifact("category_vectorizer.joblib")
        self.category_model = load_artifact("category_model.joblib")

        # dict: category -> {"vectorizer": ..., "model": ..., "text_field": ...}
        self.subcategory_models = load_artifact("subcategory_models.joblib")

        # dict: (category, subcategory) -> {"assignment_group", "confidence", "status"}
        self.routing_lookup = load_artifact("routing_lookup.joblib")

        self.search_vectorizer = load_artifact("search_vectorizer.joblib")
        self.solution_df = load_artifact("solution_df.joblib")

    # ---------------- CATEGORY ----------------
    def predict_category(self, short_description, ticket_type="", issue_request="", software_system=""):
        text = build_input_text(short_description, ticket_type, issue_request, software_system)
        vec = self.category_vectorizer.transform([text])
        return self.category_model.predict(vec)[0]

    # ---------------- SUBCATEGORY ----------------
    def predict_subcategory(self, category, short_description, ticket_type="", issue_request="", software_system=""):
        if category in RULE_BASED_SUBCATEGORY:
            return RULE_BASED_SUBCATEGORY[category]

        if category not in self.subcategory_models:
            # category has no trained subcategory model and no rule -> be honest about it
            return "NOT_AVAILABLE"

        entry = self.subcategory_models[category]
        if entry["text_field"] == "model_text":
            text = build_input_text(short_description, ticket_type, issue_request, software_system)
        else:
            text = clean_text(short_description)

        vec = entry["vectorizer"].transform([text])
        return entry["model"].predict(vec)[0]

    # ---------------- ROUTING ----------------
    def get_routing(self, category, subcategory):
        """
        IMPORTANT: 'confidence' here is NOT a machine-learning confidence
        score and NOT an SVM probability. It is the percentage of PAST
        tickets with this exact (category, subcategory) that historically
        went to this assignment group. It answers "how consistently has
        this type of ticket been routed this way before?", not "how sure
        is the model about its category/subcategory prediction?".
        """
        key = (category, subcategory)
        if key not in self.routing_lookup:
            return {"assignment_group": "GENERAL HELPDESK", "confidence": 0, "status": "MANUAL REVIEW"}
        return self.routing_lookup[key]

    # ---------------- SOLUTION RECOMMENDATION ----------------
    def recommend_solutions(self, new_ticket, predicted_category, predicted_subcategory, top_k=3):
        filtered_df = self.solution_df[
            (self.solution_df["category"] == predicted_category) &
            (self.solution_df["subcategory"] == predicted_subcategory)
        ]
        if len(filtered_df) == 0:
            return None

        from sklearn.metrics.pairwise import cosine_similarity

        filtered_tfidf = self.search_vectorizer.transform(filtered_df["search_text"])
        query_vector = self.search_vectorizer.transform([clean_text(new_ticket)])

        similarities = cosine_similarity(query_vector, filtered_tfidf).flatten()
        top_indices = similarities.argsort()[-10:][::-1]

        recs = filtered_df.iloc[top_indices].copy()
        recs["similarity"] = similarities[top_indices]
        recs = recs.drop_duplicates(subset=["number"]).head(top_k)

        return recs[["number", "short_description", "category", "subcategory", "close_notes", "similarity"]]

    # ---------------- FULL PIPELINE (same shape as the original predict_ticket) ----------------
    def predict_ticket(self, short_description, ticket_type="", issue_request="", software_system=""):
        if not short_description or not str(short_description).strip():
            raise ValueError("short_description cannot be empty")

        category = self.predict_category(short_description, ticket_type, issue_request, software_system)
        subcategory = self.predict_subcategory(category, short_description, ticket_type, issue_request, software_system)
        routing = self.get_routing(category, subcategory)
        recommendations = self.recommend_solutions(short_description, category, subcategory, top_k=3)

        return {
            "ticket": short_description,
            "category": category,
            "subcategory": subcategory,
            "assignment_group": routing["assignment_group"],
            "routing_confidence": routing["confidence"],
            "routing_status": routing["status"],
            "recommendations": recommendations,
        }
