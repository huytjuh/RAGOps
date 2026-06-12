from __future__ import annotations

import math
from dataclasses import dataclass

from _02_transcripts._01_preprocessing import Conversation, ConversationTurn


@dataclass(frozen=True)
class IntentPrediction:
    conversation_id: str
    turn_id: int | None
    scope: str
    intent: str
    confidence: float
    probabilities: dict[str, float]
    triggers: list[str]


class TranscriptIntentRecognizer:
    """Recognize intent for each customer turn and the whole conversation."""

    DEFAULT_INTENTS = [
        "billing_help",
        "technical_support",
        "account_change",
        "cancellation_risk",
        "delivery_status",
        "complaint_escalation",
        "product_question",
    ]
    KEYWORDS = {
        "billing_help": {"bill", "charged", "refund", "invoice", "fee", "payment"},
        "technical_support": {"error", "login", "password", "bug", "app", "crash", "broken"},
        "account_change": {"address", "email", "plan", "upgrade", "downgrade", "account"},
        "cancellation_risk": {"cancel", "leaving", "switch", "competitor", "close"},
        "delivery_status": {"delivery", "shipment", "tracking", "late", "arrive", "package"},
        "complaint_escalation": {"manager", "escalate", "complaint", "angry", "ignored"},
        "product_question": {"feature", "how", "where", "can", "does", "support"},
    }
    TRIGGER_TERMS = {
        "churn_risk": {"cancel", "leaving", "switch"},
        "repeat_contact": {"again", "still", "third", "multiple"},
        "supervisor_request": {"manager", "supervisor", "escalate"},
        "financial_remediation": {"refund", "charged", "fee"},
    }

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        intents: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.intents = intents or self.DEFAULT_INTENTS
        self._pipeline = None

    def predict_customer_turns(self, conversation: Conversation) -> list[IntentPrediction]:
        return [self.predict_turn(turn) for turn in conversation.customer_turns]

    def predict_conversation(self, conversation: Conversation) -> IntentPrediction:
        probabilities = self._probabilities(conversation.text)
        intent = max(probabilities, key=probabilities.get)
        return IntentPrediction(
            conversation_id=conversation.conversation_id,
            turn_id=None,
            scope="conversation",
            intent=intent,
            confidence=round(probabilities[intent], 4),
            probabilities=probabilities,
            triggers=self.detect_triggers(conversation.text),
        )

    def predict_turn(self, turn: ConversationTurn) -> IntentPrediction:
        probabilities = self._probabilities(turn.text)
        intent = max(probabilities, key=probabilities.get)
        return IntentPrediction(
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
            scope="customer_turn",
            intent=intent,
            confidence=round(probabilities[intent], 4),
            probabilities=probabilities,
            triggers=self.detect_triggers(turn.text),
        )

    def detect_triggers(self, text: str) -> list[str]:
        lowered = text.lower()
        return [
            trigger
            for trigger, terms in self.TRIGGER_TERMS.items()
            if any(term in lowered for term in terms)
        ]

    def _probabilities(self, text: str) -> dict[str, float]:
        transformer_prediction = self._predict_with_transformers(text)
        if transformer_prediction is not None:
            return transformer_prediction

        tokens = set(text.lower().split())
        scores = {
            intent: 1.0 + len(tokens.intersection(self.KEYWORDS.get(intent, set())))
            for intent in self.intents
        }
        total = sum(math.exp(score) for score in scores.values())
        return {intent: round(math.exp(score) / total, 4) for intent, score in scores.items()}

    def _predict_with_transformers(self, text: str) -> dict[str, float] | None:
        try:
            from transformers import pipeline
        except ImportError:
            return None

        if self._pipeline is None:
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                hypothesis_template="This customer turn is about {}.",
            )
        result = self._pipeline(text, candidate_labels=self.intents, multi_label=False)
        labels = [str(label) for label in result["labels"]]
        scores = [float(score) for score in result["scores"]]
        return dict(zip(labels, scores, strict=True))
