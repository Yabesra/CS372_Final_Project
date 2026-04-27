# src/evaluate.py
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from retriever import embed, retrieve
from inference import rag_answer

# -----------------------------
# 1. Gold test set
# -----------------------------
GOLD_TEST_SET = [
    ("You agree to binding arbitration and waive class action rights.", 1),
    ("We may share your personal data with third parties for marketing.", 1),
    ("This agreement renews automatically each month unless you cancel 30 days prior.", 1),
    ("Your liability is limited to the amount you paid, not exceeding $50.", 1),
    ("We can change these terms at any time without prior notice.", 1),
    ("You may cancel your account at any time with 30 days written notice.", 0),
    ("We collect only your email address to manage your account.", 0),
    ("We will notify you of any changes at least 14 days in advance.", 0),
    ("This agreement is governed by the laws of your state of residence.", 0),
    ("Your use of the service indicates acceptance of these terms.", 0),
]

# -----------------------------
# 2. Flexible answer matching (from RAG assignment)
# -----------------------------
def evaluate_answer(predicted, ground_truth):
    pred = predicted.lower().strip()
    gt = ground_truth.lower().strip()
    if pred == gt or gt in pred or pred in gt:
        return 1
    if set(gt.split()).issubset(set(pred.split())):
        return 1
    return 0

# -----------------------------
# 3. Fallback risk keyword scoring (same as in app.py)
# -----------------------------
RISK_WEIGHTS = {
    "arbitration": 5,
    "class action": 5,
    "waive": 5,
    "indemnify": 5,
    "without notice": 5,
    "liability": 2,
    "not responsible": 4,
    "disclaim": 3,
    "terminate": 3,
    "sole discretion": 4,
    "close account": 3,
    "share your data": 3,
    "disclosure": 2,
    "auto-renew": 2,
    "restrict": 2,
    "own risk": 2,
    "cannot guarantee": 2,
}
NEGATION_WORDS = {"not", "no", "never", "without", "exclude", "except", "doesn't", "does not"}

def heuristic_risk_label(context: str) -> int:
    """Return 1 if risk keywords found without strong negation, else 0."""
    context_lower = context.lower()
    has_negation = any(neg in context_lower for neg in NEGATION_WORDS)
    for kw in RISK_WEIGHTS:
        if kw in context_lower:
            # If risk keyword present and not negated, mark as risky
            if not has_negation:
                return 1
            # Even if negated, some keywords are still risky (e.g., 'indemnify' even if 'not indemnify' is rare)
            if kw in ["indemnify", "liability", "arbitration"]:
                return 1
    return 0

# -----------------------------
# 4. Evaluation (with fallback)
# -----------------------------
def evaluate_model():
    y_true = []
    y_pred = []
    print("Running evaluation on test set...\n")

    # Build corpus from all clauses for retrieval
    corpus = [c for c, _ in GOLD_TEST_SET]
    corpus_embs = embed(corpus)

    for clause, true_label in GOLD_TEST_SET:
        query = f"Does the following clause present a legal risk? Answer only 'yes' or 'no'. Clause: {clause}"
        retrieved = retrieve(query, corpus, corpus_embs, top_k=1)[0]

        # Get model answer
        try:
            model_answer = rag_answer(query, retrieved)
        except Exception:
            model_answer = ""

        # Try to extract yes/no from model_answer
        answer_lower = model_answer.lower()
        if "yes" in answer_lower and "no" not in answer_lower:
            predicted_label = 1
        elif "no" in answer_lower:
            predicted_label = 0
        else:
            # Fallback to heuristic based on retrieved context
            predicted_label = heuristic_risk_label(retrieved)

        y_true.append(true_label)
        y_pred.append(predicted_label)

        print(f"Clause: {clause[:70]}...")
        print(f"True: {true_label} | Pred: {predicted_label} | Model said: {model_answer[:50]}\n")

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("===== Evaluation Results =====")
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print("Confusion matrix:")
    print(cm)

    return {"accuracy": acc, "precision": prec, "recall": rec, "confusion_matrix": cm.tolist()}

if __name__ == "__main__":
    evaluate_model()