from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentTopicPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    triggers: list[str]


class DocumentTopicClassifier:
    """Classify document purpose and business risk from extracted text."""

    DEFAULT_TOPICS = [
        "invoice",
        "contract",
        "support_case",
        "policy",
        "compliance_notice",
        "customer_correspondence",
    ]
    KEYWORDS = {
        "invoice": {"invoice", "amount", "due", "payment", "subtotal", "tax", "balance"},
        "contract": {"agreement", "term", "party", "signature", "effective", "clause"},
        "support_case": {"case", "issue", "resolved", "support", "ticket", "incident"},
        "policy": {"policy", "procedure", "guideline", "scope", "responsibility"},
        "compliance_notice": {"breach", "regulatory", "audit", "compliance", "violation"},
        "customer_correspondence": {"dear", "sincerely", "customer", "regards", "request"},
    }
    TRIGGER_TERMS = {
        "payment_due": {"overdue", "balance", "due"},
        "compliance_risk": {"breach", "violation", "regulatory"},
        "contractual_obligation": {"agreement", "clause", "termination"},
        "customer_escalation": {"urgent", "complaint", "escalate"},
    }

    def __init__(self, topics: list[str] | None = None) -> None:
        self.topics = topics or self.DEFAULT_TOPICS

    def predict(self, text: str) -> DocumentTopicPrediction:
        tokens = set(text.lower().split())
        scores = {
            topic: 1.0 + len(tokens.intersection(self.KEYWORDS.get(topic, set())))
            for topic in self.topics
        }
        total = sum(math.exp(score) for score in scores.values())
        probabilities = {
            topic: round(math.exp(score) / total, 4) for topic, score in scores.items()
        }
        label = max(probabilities, key=probabilities.get)
        return DocumentTopicPrediction(
            label=label,
            confidence=probabilities[label],
            probabilities=probabilities,
            triggers=self.detect_triggers(text),
        )

    def detect_triggers(self, text: str) -> list[str]:
        lowered = text.lower()
        return [
            trigger
            for trigger, terms in self.TRIGGER_TERMS.items()
            if any(term in lowered for term in terms)
        ]
