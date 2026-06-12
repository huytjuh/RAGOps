from __future__ import annotations

from dataclasses import dataclass, field

from _01_complaints._05_evaluation import ComplaintEvaluationReport


@dataclass(frozen=True)
class MonitoringDecision:
    status: str
    alerts: list[str]
    recommended_actions: list[str]
    metrics: dict[str, float | int | None] = field(default_factory=dict)


class ComplaintMonitor:
    """Turn evaluation metrics into deployment monitoring alerts."""

    def __init__(
        self,
        min_topic_coherence: float = 0.55,
        max_label_entropy: float = 0.92,
        max_trigger_rate: float = 0.35,
        min_accuracy: float = 0.75,
    ) -> None:
        self.min_topic_coherence = min_topic_coherence
        self.max_label_entropy = max_label_entropy
        self.max_trigger_rate = max_trigger_rate
        self.min_accuracy = min_accuracy

    def assess(self, report: ComplaintEvaluationReport) -> MonitoringDecision:
        alerts: list[str] = []
        actions: list[str] = []

        if report.topic_coherence < self.min_topic_coherence:
            alerts.append("topic_coherence_below_threshold")
            actions.append("Review BERTopic clusters and refresh embeddings on recent complaints.")
        if report.label_entropy > self.max_label_entropy:
            alerts.append("label_entropy_above_threshold")
            actions.append("Inspect class drift, emerging topics, and classifier calibration.")
        if report.trigger_rate > self.max_trigger_rate:
            alerts.append("trigger_rate_above_threshold")
            actions.append("Route high-risk complaints to operations and compliance queues.")
        if report.accuracy is not None and report.accuracy < self.min_accuracy:
            alerts.append("classifier_accuracy_below_threshold")
            actions.append("Retrain or fine-tune BERT with newly labeled complaint examples.")

        status = "healthy" if not alerts else "warning"
        if {"trigger_rate_above_threshold", "classifier_accuracy_below_threshold"}.issubset(alerts):
            status = "critical"

        return MonitoringDecision(
            status=status,
            alerts=alerts,
            recommended_actions=actions,
            metrics={
                "accuracy": report.accuracy,
                "macro_f1": report.macro_f1,
                "topic_coherence": report.topic_coherence,
                "label_entropy": report.label_entropy,
                "trigger_rate": report.trigger_rate,
                "sample_size": report.sample_size,
            },
        )
