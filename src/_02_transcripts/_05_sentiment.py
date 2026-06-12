from __future__ import annotations

from dataclasses import dataclass

from _02_transcripts._01_preprocessing import Conversation, ConversationTurn


@dataclass(frozen=True)
class SentimentPrediction:
    conversation_id: str
    turn_id: int
    sentiment: str
    score: float
    negative_terms: list[str]


class TurnSentimentAnalyzer:
    """Estimate sentiment for customer turns."""

    POSITIVE_TERMS = {
        "thanks",
        "thank",
        "great",
        "helpful",
        "resolved",
        "appreciate",
        "good",
    }
    NEGATIVE_TERMS = {
        "angry",
        "bad",
        "broken",
        "cancel",
        "charged",
        "frustrated",
        "ignored",
        "late",
        "refund",
        "terrible",
        "upset",
        "wrong",
    }

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> None:
        self.model_name = model_name
        self._pipeline = None

    def analyze_conversation(self, conversation: Conversation) -> list[SentimentPrediction]:
        return [self.analyze_turn(turn) for turn in conversation.customer_turns]

    def analyze_turn(self, turn: ConversationTurn) -> SentimentPrediction:
        transformer_prediction = self._analyze_with_transformers(turn.text)
        if transformer_prediction is not None:
            sentiment, score = transformer_prediction
        else:
            sentiment, score = self._lexicon_sentiment(turn.tokens)

        negative_terms = sorted(set(turn.tokens).intersection(self.NEGATIVE_TERMS))
        return SentimentPrediction(
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
            sentiment=sentiment,
            score=round(score, 4),
            negative_terms=negative_terms,
        )

    def _analyze_with_transformers(self, text: str) -> tuple[str, float] | None:
        try:
            from transformers import pipeline
        except ImportError:
            return None

        if self._pipeline is None:
            self._pipeline = pipeline("sentiment-analysis", model=self.model_name)
        result = self._pipeline(text)[0]
        label = str(result["label"]).lower()
        score = float(result["score"])
        sentiment = "positive" if "pos" in label else "negative"
        return sentiment, score

    def _lexicon_sentiment(self, tokens: list[str]) -> tuple[str, float]:
        positives = len(set(tokens).intersection(self.POSITIVE_TERMS))
        negatives = len(set(tokens).intersection(self.NEGATIVE_TERMS))
        if positives == negatives:
            return "neutral", 0.5
        if positives > negatives:
            return "positive", min(0.5 + (positives * 0.15), 0.95)
        return "negative", min(0.5 + (negatives * 0.15), 0.95)
