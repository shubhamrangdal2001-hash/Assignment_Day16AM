"""
Part D — AI-Augmented Task: Type I & II Errors in Fraud Detection
Day 16 Take-Home | PG Diploma AI-ML | IIT Gandhinagar
"""

# ── Prompt used ───────────────────────────────────────────────────────────
PROMPT = (
    "Explain Type I and Type II errors in the context of a fraud detection system. "
    "What are the real-world consequences of each, and how do you balance them?"
)

# ── AI Output (documented) ────────────────────────────────────────────────
AI_OUTPUT = """
In a fraud detection system:

Type I Error (False Positive): The model flags a legitimate transaction as fraudulent.
Real-world consequence: A genuine customer gets their card blocked or purchase declined.
This hurts user experience, increases support calls, and can lead to customer churn.

Type II Error (False Negative): The model misses an actual fraud transaction.
Real-world consequence: A fraudulent charge goes through. The bank or merchant
absorbs the loss, and the customer may lose money or trust.

Balancing them:
This is exactly the precision-recall tradeoff. If you lower the decision threshold,
you catch more fraud (higher recall) but also block more legit transactions (lower precision).
If you raise the threshold, fewer legit users are bothered, but more fraud slips through.
The right balance depends on the cost ratio: how much does a missed fraud cost vs
how much does a false block cost in customer lifetime value? Typically, financial
institutions monitor F-beta score with beta > 1 when recall (catching fraud) matters more.
"""

# ── My evaluation ────────────────────────────────────────────────────────
EVALUATION = """
Does the AI correctly explain the precision-recall tradeoff as the Type I/II error tradeoff?

YES — it does. The AI correctly maps:
  - Type I  error ↔ False Positive ↔ Low Precision (flagging legit as fraud)
  - Type II error ↔ False Negative ↔ Low Recall    (missing real fraud)

And it explains the tradeoff: lowering the decision threshold increases recall but
hurts precision, which is exactly the precision-recall tradeoff.

One gap: the AI mentions F-beta but doesn't explicitly say that
"raising recall = reducing Type II errors = accepting more Type I errors."
That link could be made clearer. Overall, the explanation is accurate and business-relevant.
"""

if __name__ == "__main__":
    print("=" * 55)
    print("Part D — AI-Augmented Task")
    print("=" * 55)
    print("\n[Prompt Used]")
    print(PROMPT)
    print("\n[AI Output]")
    print(AI_OUTPUT)
    print("\n[My Evaluation]")
    print(EVALUATION)
