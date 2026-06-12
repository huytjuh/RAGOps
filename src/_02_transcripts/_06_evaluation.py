from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from _02_transcripts._03_topic_classifier import IntentPrediction
from _02_transcripts._04_topic_discovery import DiscoveredIntent
from _02_transcripts._05_sentiment import SentimentPrediction


@dataclass(frozen=True)
class LLMJudgeEvaluation:
    helpfulness: float
    completeness: float
    groundedness: float
    hallucination_risk: float
    overall: float
    issues: list[str]


@dataclass(frozen=True)
class ConversationEvaluationReport:
    intent_accuracy: float | None
    intent_macro_f1: float | None
    discovery_coherence: float
    intent_entropy: float
    negative_sentiment_rate: float
    trigger_rate: float
    judge: LLMJudgeEvaluation | None
    sample_size: int


class QueryResponseJudge:
    """Score response quality and hallucination risk against retrieved evidence."""

    HALLUCINATION_MARKERS = {"guaranteed", "always", "never", "definitely", "no evidence needed"}

    def evaluate(
        self,
        query: str,
        response: str,
        evidence: list[str],
        required_points: list[str] | None = None,
    ) -> LLMJudgeEvaluation:
        query_terms = self._content_terms(query)
        response_terms = self._content_terms(response)
        evidence_terms = self._content_terms(" ".join(evidence))
        required_terms = self._content_terms(" ".join(required_points or []))

        helpfulness = self._overlap_score(query_terms, response_terms)
        completeness = self._overlap_score(required_terms or query_terms, response_terms)
        groundedness = self._overlap_score(response_terms, evidence_terms)
        hallucination_risk = 1.0 - groundedness

        issues = []
        if helpfulness < 0.45:
            issues.append("low_helpfulness")
        if completeness < 0.45:
            issues.append("incomplete_answer")
        if groundedness < 0.55:
            issues.append("weak_evidence_grounding")
        if any(marker in response.lower() for marker in self.HALLUCINATION_MARKERS):
            hallucination_risk = min(1.0, hallucination_risk + 0.2)
            issues.append("overconfident_language")

        overall = (helpfulness * 0.3) + (completeness * 0.3) + (groundedness * 0.4)
        return LLMJudgeEvaluation(
            helpfulness=round(helpfulness, 4),
            completeness=round(completeness, 4),
            groundedness=round(groundedness, 4),
            hallucination_risk=round(hallucination_risk, 4),
            overall=round(overall, 4),
            issues=issues,
        )

    def _content_terms(self, text: str) -> set[str]:
        stopwords = {"the", "and", "for", "with", "that", "this", "you", "your", "are"}
        return {
            token.strip(".,!?;:").lower()
            for token in text.split()
            if len(token.strip(".,!?;:")) > 2 and token.lower() not in stopwords
        }

    def _overlap_score(self, expected: set[str], actual: set[str]) -> float:
        if not expected:
            return 1.0
        return min(len(expected.intersection(actual)) / len(expected), 1.0)


class TranscriptEvaluator:
    """Evaluate intent classification, discovery health, sentiment, and judge scores."""

    def evaluate(
        self,
        predictions: list[IntentPrediction],
        discovered_intents: list[DiscoveredIntent],
        sentiments: list[SentimentPrediction],
        truth: dict[tuple[str, int | None], str] | None = None,
        judge: LLMJudgeEvaluation | None = None,
    ) -> ConversationEvaluationReport:
        return ConversationEvaluationReport(
            intent_accuracy=self.intent_accuracy(predictions, truth) if truth else None,
            intent_macro_f1=self.intent_macro_f1(predictions, truth) if truth else None,
            discovery_coherence=self.discovery_coherence(discovered_intents),
            intent_entropy=self.intent_entropy(predictions),
            negative_sentiment_rate=self.negative_sentiment_rate(sentiments),
            trigger_rate=self.trigger_rate(predictions),
            judge=judge,
            sample_size=len({prediction.conversation_id for prediction in predictions}),
        )

    def intent_accuracy(
        self,
        predictions: list[IntentPrediction],
        truth: dict[tuple[str, int | None], str],
    ) -> float:
        matched = [
            prediction
            for prediction in predictions
            if (prediction.conversation_id, prediction.turn_id) in truth
        ]
        if not matched:
            return 0.0
        correct = sum(
            prediction.intent == truth[(prediction.conversation_id, prediction.turn_id)]
            for prediction in matched
        )
        return round(correct / len(matched), 4)

    def intent_macro_f1(
        self,
        predictions: list[IntentPrediction],
        truth: dict[tuple[str, int | None], str],
    ) -> float:
        labels = sorted({prediction.intent for prediction in predictions}.union(truth.values()))
        if not labels:
            return 0.0
        scores = []
        for label in labels:
            tp = sum(
                prediction.intent == label
                and truth.get((prediction.conversation_id, prediction.turn_id)) == label
                for prediction in predictions
            )
            fp = sum(
                prediction.intent == label
                and truth.get((prediction.conversation_id, prediction.turn_id)) != label
                for prediction in predictions
            )
            fn = sum(
                prediction.intent != label
                and truth.get((prediction.conversation_id, prediction.turn_id)) == label
                for prediction in predictions
            )
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            scores.append(f1)
        return round(sum(scores) / len(scores), 4)

    def discovery_coherence(self, discovered_intents: list[DiscoveredIntent]) -> float:
        if not discovered_intents:
            return 0.0
        scores = []
        for intent in discovered_intents:
            unique_keyword_ratio = len(set(intent.keywords)) / max(len(intent.keywords), 1)
            support = min(intent.size / 5, 1.0)
            scores.append((unique_keyword_ratio * 0.7) + (support * 0.3))
        return round(sum(scores) / len(scores), 4)

    def intent_entropy(self, predictions: list[IntentPrediction]) -> float:
        if not predictions:
            return 0.0
        counts = Counter(prediction.intent for prediction in predictions)
        total = len(predictions)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        return max(0.0, round(entropy / max_entropy, 4))

    def negative_sentiment_rate(self, sentiments: list[SentimentPrediction]) -> float:
        if not sentiments:
            return 0.0
        negative = sum(sentiment.sentiment == "negative" for sentiment in sentiments)
        return round(negative / len(sentiments), 4)

    def trigger_rate(self, predictions: list[IntentPrediction]) -> float:
        if not predictions:
            return 0.0
        triggered = sum(bool(prediction.triggers) for prediction in predictions)
        return round(triggered / len(predictions), 4)
