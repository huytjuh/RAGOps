from __future__ import annotations

from dataclasses import dataclass, field

from _02_transcripts._06_evaluation import ConversationEvaluationReport


@dataclass(frozen=True)
class ConversationMonitoringDecision:
    status: str
    alerts: list[str]
    recommended_actions: list[str]
    metrics: dict[str, float | int | None] = field(default_factory=dict)


class ConversationMonitor:
    """Monitor conversational analytics quality and operational risk."""

    def __init__(
        self,
        min_discovery_coherence: float = 0.55,
        max_intent_entropy: float = 0.94,
        max_negative_sentiment_rate: float = 0.45,
        max_trigger_rate: float = 0.4,
        max_hallucination_risk: float = 0.35,
        min_judge_overall: float = 0.65,
        min_intent_accuracy: float = 0.75,
    ) -> None:
        self.min_discovery_coherence = min_discovery_coherence
        self.max_intent_entropy = max_intent_entropy
        self.max_negative_sentiment_rate = max_negative_sentiment_rate
        self.max_trigger_rate = max_trigger_rate
        self.max_hallucination_risk = max_hallucination_risk
        self.min_judge_overall = min_judge_overall
        self.min_intent_accuracy = min_intent_accuracy

    def assess(self, report: ConversationEvaluationReport) -> ConversationMonitoringDecision:
        alerts: list[str] = []
        actions: list[str] = []

        if report.discovery_coherence < self.min_discovery_coherence:
            alerts.append("intent_discovery_coherence_below_threshold")
            actions.append("Review emerging intent clusters and refresh transcript embeddings.")
        if report.intent_entropy > self.max_intent_entropy:
            alerts.append("intent_entropy_above_threshold")
            actions.append("Inspect new demand drivers and retrain intent taxonomy mapping.")
        if report.negative_sentiment_rate > self.max_negative_sentiment_rate:
            alerts.append("negative_sentiment_rate_above_threshold")
            actions.append("Route affected queues to service recovery and coaching workflows.")
        if report.trigger_rate > self.max_trigger_rate:
            alerts.append("conversation_trigger_rate_above_threshold")
            actions.append("Escalate churn, repeat-contact, and refund-risk conversations.")
        if report.intent_accuracy is not None and report.intent_accuracy < self.min_intent_accuracy:
            alerts.append("intent_accuracy_below_threshold")
            actions.append("Fine-tune or relabel the intent classifier with recent conversations.")
        if report.judge is not None:
            if report.judge.hallucination_risk > self.max_hallucination_risk:
                alerts.append("hallucination_risk_above_threshold")
                actions.append("Tighten retrieval grounding and require cited evidence.")
            if report.judge.overall < self.min_judge_overall:
                alerts.append("llm_response_quality_below_threshold")
                actions.append("Review prompt, retrieval coverage, and answer policy.")

        status = "healthy" if not alerts else "warning"
        critical_alerts = {
            "hallucination_risk_above_threshold",
            "intent_accuracy_below_threshold",
            "negative_sentiment_rate_above_threshold",
        }
        if len(critical_alerts.intersection(alerts)) >= 2:
            status = "critical"

        judge = report.judge
        return ConversationMonitoringDecision(
            status=status,
            alerts=alerts,
            recommended_actions=actions,
            metrics={
                "intent_accuracy": report.intent_accuracy,
                "intent_macro_f1": report.intent_macro_f1,
                "discovery_coherence": report.discovery_coherence,
                "intent_entropy": report.intent_entropy,
                "negative_sentiment_rate": report.negative_sentiment_rate,
                "trigger_rate": report.trigger_rate,
                "judge_overall": judge.overall if judge else None,
                "hallucination_risk": judge.hallucination_risk if judge else None,
                "sample_size": report.sample_size,
            },
        )
