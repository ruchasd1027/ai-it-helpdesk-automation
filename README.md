# AI-Based IT Helpdesk Ticket Triage and Solution Recommendation System

## 1. Project structure

```
helpdesk-ai/
├── data/
│   └── synthetic-it-call-center-tickets.csv   <- put your dataset here
├── models/                                     <- created automatically by training
│   ├── category_vectorizer.joblib
│   ├── category_model.joblib                   (SVM, used by the app)
│   ├── category_model_logreg.joblib             (Logistic Regression, kept for comparison/report)
│   ├── subcategory_models.joblib
│   ├── routing_lookup.joblib
│   ├── search_vectorizer.joblib
│   ├── solution_df.joblib
│   └── evaluation_results.json                  <- all metrics below come from this file
├── helpdesk_core.py     <- shared logic (text cleaning, prediction, routing, recommendation)
├── train_models.py      <- trains everything and saves it to models/
├── app.py                <- Gradio demo, loads models/, never retrains
├── requirements.txt
└── README.md
```

## 2. Windows / VS Code terminal commands

Open the `helpdesk-ai` folder in VS Code, open a terminal (`` Ctrl+` ``), then:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Put your CSV at `data\synthetic-it-call-center-tickets.csv`, then:

```bat
python train_models.py
```

This prints every metric below and saves all models into `models\`. Then:

```bat
python app.py
```

This opens the Gradio app in your browser (no retraining happens here — it just loads the files `train_models.py` created).

## 3. Problems found in the original code, and what was fixed

| # | Problem | Fix |
|---|---|---|
| 1 | `predict_subcategory()` used `printer_model`, `network_model`, `account_model`, etc. before those variables existed in the notebook — it only worked by accident, because cells happened to run top-to-bottom once. | All models are now trained once in `train_models.py`, saved to disk, and loaded by name in `helpdesk_core.py`. There is no "variable defined later" problem anymore. |
| 2 | No model saving — the Gradio app depended entirely on the notebook's memory. Restarting Python meant retraining from scratch. | `joblib.dump()` / `joblib.load()` for every model and vectorizer. `app.py` starts in under a second. |
| 3 | `EMAIL` and `HARDWARE` subcategories were hardcoded (`"REQUEST"`, `"MANUAL REVIEW"`) without explanation. | Kept — because the data genuinely supports it (see Limitations) — but now documented explicitly as a deliberate rule in `helpdesk_core.RULE_BASED_SUBCATEGORY`, not a silent shortcut. |
| 4 | `tfidf` variable was reused and overwritten for two different feature sets (`clean_text`, then `model_text`), which works only if cells run in the right order. | Each model now has its own clearly named vectorizer, saved separately. No shared/overwritten variables. |
| 5 | No systematic evaluation beyond accuracy; per-class performance and class imbalance were never reported. | `train_models.py` now prints and saves accuracy, macro precision/recall/F1, weighted F1, a full classification report, and a confusion matrix for every model. |
| 6 | Only one model (SVM) was evaluated — no real comparison. | Added `LogisticRegression` trained and evaluated on the exact same train/test split as the SVM, for a fair, like-for-like comparison. |

**No working functionality was removed.** TF-IDF + SVM category prediction, per-category subcategory prediction, routing with confidence/status, and top-3 historical solution recommendations all work exactly as before — they're just no longer held together by notebook cell order.

## 4. SVM vs Logistic Regression — actual measured results

Both models were trained on the same 80/20 stratified split (`random_state=42`) of your `synthetic-it-call-center-tickets.csv` (27,599 usable rows), using the same TF-IDF features. These are the real numbers from `models/evaluation_results.json` — nothing here is estimated.

| Metric | Linear SVM (primary) | Logistic Regression |
|---|---|---|
| Accuracy | **0.9582** | 0.9109 |
| Precision (macro) | **0.8646** | 0.6844 |
| Recall (macro) | 0.8900 | **0.9056** |
| F1 (macro) | **0.8737** | 0.7518 |
| F1 (weighted) | **0.9618** | 0.9286 |

**Honest finding:** SVM clearly wins here — higher on every metric except macro recall. Logistic Regression tends to recall the small classes (ACCOUNT, HARDWARE, NETWORK) more often, but at the cost of far more false positives — its precision on ACCOUNT, for example, is only 0.33, meaning 2 out of 3 tickets it labels ACCOUNT are actually something else. Say in your report: *"We compared SVM against Logistic Regression on the same held-out test set; SVM had clearly better accuracy and macro-F1, so we kept it as the primary model."* This is a clean, real result — no hedging needed this time.

## 5. Model saving and reloading — how it works

`train_models.py` calls `joblib.dump(model, "models/xyz.joblib")` right after training each model or vectorizer. `joblib` is the standard way to save scikit-learn objects (plain `pickle` would also work, but `joblib` handles NumPy arrays more efficiently, which TF-IDF matrices and model weights are full of).

`app.py` never imports anything from `train_models.py`. It only calls `HelpdeskModels()` from `helpdesk_core.py`, which does `joblib.load()` for every file in `models/`. This was verified in a **separate, fresh Python process** (not the one that did the training) — predictions worked correctly with no retraining.

If `models/` is missing or incomplete, `HelpdeskModels()` raises a clear `FileNotFoundError` telling you to run `train_models.py` first, instead of crashing with a confusing `NameError` deep inside a prediction.

## 6. Important limitations (say these out loud in your viva — they make you look stronger, not weaker)

**Class imbalance.** Of 27,599 tickets, 88.5% are `SOFTWARE`. `HARDWARE` has only 38 total rows and `NETWORK` only 83. Their test sets end up with just 8 and 17 examples, so precision/recall numbers for those two categories are not statistically reliable — say this explicitly rather than presenting them as solid.

**ACCOUNT and SECURITY genuinely overlap with SOFTWARE in the data.** The confusion matrix shows 145 real `SOFTWARE` tickets predicted as `ACCOUNT`, and tickets like *"Request for additional [X] licenses"* appear near-identically labeled as both `ACCOUNT` and `SOFTWARE/ACCESS`. This is why `ACCOUNT` precision is only 0.52 (SVM) — it is a data-labeling ambiguity, not something a better algorithm fixes by itself.

**The majority class pulls ambiguous tickets toward itself.** When I tested the reloaded model on informal, everyday tickets it had never seen, two were genuinely misclassified:
- *"The whole office network is down and nobody can connect."* → predicted `SOFTWARE` (should be `NETWORK`)
- *"laptop screen cracked need replacement"* → predicted `SOFTWARE` (should be `HARDWARE`)

This happens because `SOFTWARE` is 88.5% of the training data, so when a ticket's wording doesn't strongly match the vocabulary of a small category, the model defaults toward the category it has seen the most. This is a real, verified limitation (not hypothetical) and a good thing to mention and propose a fix for: collecting more `NETWORK`/`HARDWARE` examples, or using `class_weight="balanced"` more aggressively, would help — both are already partly applied here, but can't fully overcome an 88%-vs-0.3% imbalance.

**Routing confidence is not a prediction probability.** It is the percentage of historical tickets with that exact (category, subcategory) that were routed to a given team — e.g. 81.8% means "81.8% of past NETWORK/MALFUNCTION tickets went to the Network Team," not "the model is 81.8% sure."

## 7. What was actually run and verified vs. what you should still check yourself

**Actually executed in this environment, with real output:**
- Full training run on your 27,602-row CSV (shown in full above).
- A fresh second Python process loading only the saved files and running 8 test tickets, including informal/everyday phrasing and the two misclassifications described above.
- Empty-string and whitespace-only ticket descriptions, confirmed to fail safely with a clear message instead of crashing.
- Deleting the `models/` folder and confirming `HelpdeskModels()` raises a clear, readable error instead of a cryptic one.
- Building the Gradio `Blocks` app object and confirming it constructs without error.

**You should verify yourself (needs a GUI/browser, which this environment doesn't have):**
- Actually clicking "Analyze Ticket" and "Clear All" in the running Gradio window on your machine.
- That `python app.py` opens a browser tab correctly on Windows.

## 8. Simple explanation of the workflow, for your viva

1. **Training (`train_models.py`, run once):** reads the CSV → cleans the text → splits 80% train / 20% test → turns text into TF-IDF numbers *using only the training rows* → trains an SVM (and an ANN for comparison) to predict ticket category → trains a separate small SVM per category to predict subcategory → builds a lookup table of which team handled similar tickets historically → builds a searchable index of past resolved tickets for recommendations → saves every one of these pieces to the `models/` folder.
2. **App (`app.py`, run any time after):** loads those saved pieces from disk (instant, no training) → when someone types a ticket, it cleans the text the same way, predicts category, predicts subcategory using that category's specific model, looks up which team usually handles that combination, and finds the 3 most similar past resolved tickets by cosine similarity → shows all of it in the Gradio interface.
3. The key engineering idea, if asked: **training and serving are separate stages.** You train once, save the result, and the live app only ever *uses* the saved result — it never retrains itself, which is why it's both fast and reliable.
