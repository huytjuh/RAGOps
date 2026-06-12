from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ComplaintPrediction:
    complaint_id: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    severity: str
    triggers: list[str]


class BertComplaintClassifier:
    """BERT complaint classifier with a rules fallback for local smoke tests."""

    DEFAULT_LABELS = [
        "billing",
        "service_quality",
        "technical_issue",
        "delivery_delay",
        "compliance_risk",
    ]

    KEYWORDS = {
        "billing": {"bill", "billing", "charged", "refund", "invoice", "fee", "payment"},
        "service_quality": {"rude", "agent", "support", "ignored", "unhelpful", "call"},
        "technical_issue": {"error", "bug", "crash", "login", "password", "app", "website"},
        "delivery_delay": {"late", "delay", "delivery", "shipment", "missing", "arrived"},
        "compliance_risk": {"fraud", "legal", "privacy", "breach", "unauthorized", "regulator"},
    }
    HIGH_SEVERITY_TERMS = {"fraud", "breach", "unauthorized", "legal", "regulator"}

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        labels: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.labels = labels or self.DEFAULT_LABELS
        self._pipeline = None

    def predict(self, complaint_id: str, text: str) -> ComplaintPrediction:
        bert_prediction = self._predict_with_transformers(text)
        if bert_prediction is not None:
            label, probabilities = bert_prediction
        else:
            probabilities = self._keyword_probabilities(text)
            label = max(probabilities, key=probabilities.get)

        triggers = self.detect_triggers(text)
        severity = self._severity(text, probabilities[label])
        return ComplaintPrediction(
            complaint_id=complaint_id,
            label=label,
            confidence=round(probabilities[label], 4),
            probabilities=probabilities,
            severity=severity,
            triggers=triggers,
        )

    def predict_many(self, complaints: list[tuple[str, str]]) -> list[ComplaintPrediction]:
        return [self.predict(complaint_id, text) for complaint_id, text in complaints]

    def detect_triggers(self, text: str) -> list[str]:
        lowered = text.lower()
        triggers = []
        if any(term in lowered for term in self.HIGH_SEVERITY_TERMS):
            triggers.append("regulatory_or_legal_escalation")
        if any(term in lowered for term in {"refund", "charged", "fee"}):
            triggers.append("financial_remediation")
        if any(term in lowered for term in {"again", "still", "third", "multiple"}):
            triggers.append("repeat_contact_risk")
        if any(term in lowered for term in {"cancel", "leaving", "switch"}):
            triggers.append("churn_risk")
        return triggers

    def _predict_with_transformers(self, text: str) -> tuple[str, dict[str, float]] | None:
        try:
            from transformers import pipeline
        except ImportError:
            return None

        if self._pipeline is None:
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                hypothesis_template="This complaint is about {}.",
            )
        result = self._pipeline(text, candidate_labels=self.labels, multi_label=False)
        labels = [str(label) for label in result["labels"]]
        scores = [float(score) for score in result["scores"]]
        probabilities = dict(zip(labels, scores, strict=True))
        return labels[0], probabilities

    def _keyword_probabilities(self, text: str) -> dict[str, float]:
        tokens = set(text.lower().split())
        scores = {
            label: 1.0 + len(tokens.intersection(self.KEYWORDS.get(label, set())))
            for label in self.labels
        }
        total = sum(math.exp(score) for score in scores.values())
        return {label: round(math.exp(score) / total, 4) for label, score in scores.items()}

    def _severity(self, text: str, confidence: float) -> str:
        lowered = text.lower()
        if any(term in lowered for term in self.HIGH_SEVERITY_TERMS):
            return "critical"
        if confidence >= 0.65:
            return "high"
        if confidence >= 0.45:
            return "medium"
        return "low"
