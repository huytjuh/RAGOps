from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from _01_complaints._03_topic_classifier import ComplaintPrediction
from _01_complaints._04_topic_discovery import DiscoveredTopic


@dataclass(frozen=True)
class ComplaintEvaluationReport:
    accuracy: float | None
    macro_f1: float | None
    topic_coherence: float
    label_entropy: float
    trigger_rate: float
    sample_size: int


class ComplaintEvaluator:
    """Evaluate classifier quality, topic health, entropy, and trigger frequency."""

    def evaluate(
        self,
        predictions: list[ComplaintPrediction],
        topics: list[DiscoveredTopic],
        truth: dict[str, str] | None = None,
    ) -> ComplaintEvaluationReport:
        accuracy = self.accuracy(predictions, truth) if truth else None
        macro_f1 = self.macro_f1(predictions, truth) if truth else None
        return ComplaintEvaluationReport(
            accuracy=accuracy,
            macro_f1=macro_f1,
            topic_coherence=self.topic_coherence(topics),
            label_entropy=self.label_entropy(predictions),
            trigger_rate=self.trigger_rate(predictions),
            sample_size=len(predictions),
        )

    def accuracy(
        self,
        predictions: list[ComplaintPrediction],
        truth: dict[str, str],
    ) -> float:
        matched = [prediction for prediction in predictions if prediction.complaint_id in truth]
        if not matched:
            return 0.0
        correct = sum(prediction.label == truth[prediction.complaint_id] for prediction in matched)
        return round(correct / len(matched), 4)

    def macro_f1(
        self,
        predictions: list[ComplaintPrediction],
        truth: dict[str, str],
    ) -> float:
        labels = sorted({prediction.label for prediction in predictions}.union(truth.values()))
        if not labels:
            return 0.0

        scores = []
        for label in labels:
            tp = sum(
                prediction.label == label and truth.get(prediction.complaint_id) == label
                for prediction in predictions
            )
            fp = sum(
                prediction.label == label and truth.get(prediction.complaint_id) != label
                for prediction in predictions
            )
            fn = sum(
                prediction.label != label and truth.get(prediction.complaint_id) == label
                for prediction in predictions
            )
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            scores.append(f1)
        return round(sum(scores) / len(scores), 4)

    def topic_coherence(self, topics: list[DiscoveredTopic]) -> float:
        if not topics:
            return 0.0
        scores = []
        for topic in topics:
            unique_keywords = len(set(topic.keywords))
            density = unique_keywords / max(len(topic.keywords), 1)
            support = min(topic.size / 5, 1.0)
            scores.append((density * 0.7) + (support * 0.3))
        return round(sum(scores) / len(scores), 4)

    def label_entropy(self, predictions: list[ComplaintPrediction]) -> float:
        if not predictions:
            return 0.0
        counts = Counter(prediction.label for prediction in predictions)
        total = len(predictions)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        return max(0.0, round(entropy / max_entropy, 4))

    def trigger_rate(self, predictions: list[ComplaintPrediction]) -> float:
        if not predictions:
            return 0.0
        triggered = sum(bool(prediction.triggers) for prediction in predictions)
        return round(triggered / len(predictions), 4)
