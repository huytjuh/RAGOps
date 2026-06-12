from __future__ import annotations

from dataclasses import dataclass

from _02_transcripts._03_topic_classifier import IntentPrediction
from _02_transcripts._05_sentiment import SentimentPrediction
from _02_transcripts._07_monitoring import ConversationMonitoringDecision


@dataclass(frozen=True)
class ConversationKPIImpact:
    conversation_id: str
    intent: str
    affected_kpis: dict[str, float]
    owner: str
    priority: str


class ConversationKPIMapper:
    """Link conversation analytics outputs to contact-center and RAG KPIs."""

    KPI_WEIGHTS = {
        "billing_help": {"refund_cost": 0.8, "first_contact_resolution": 0.6, "csat": 0.5},
        "technical_support": {"digital_success_rate": 0.9, "handle_time": 0.5, "csat": 0.5},
        "account_change": {"self_service_deflection": 0.7, "handle_time": 0.4, "csat": 0.4},
        "cancellation_risk": {"churn_risk": 1.0, "retention_save_rate": 0.9, "csat": 0.6},
        "delivery_status": {"sla_attainment": 0.8, "repeat_contact_rate": 0.6, "csat": 0.5},
        "complaint_escalation": {"escalation_rate": 0.9, "agent_quality": 0.7, "csat": 0.7},
        "product_question": {"conversion_rate": 0.5, "self_service_deflection": 0.5, "csat": 0.4},
    }
    OWNERS = {
        "billing_help": "Finance Operations",
        "technical_support": "Digital Support",
        "account_change": "Customer Operations",
        "cancellation_risk": "Retention",
        "delivery_status": "Fulfillment Operations",
        "complaint_escalation": "Customer Care Leadership",
        "product_question": "Product Operations",
    }

    def map_prediction(
        self,
        prediction: IntentPrediction,
        sentiments: list[SentimentPrediction],
    ) -> ConversationKPIImpact:
        weights = self.KPI_WEIGHTS.get(prediction.intent, {"csat": 0.4})
        sentiment_multiplier = self._sentiment_multiplier(prediction.conversation_id, sentiments)
        trigger_multiplier = 1.25 if prediction.triggers else 1.0
        affected_kpis = {
            kpi: round(
                weight * prediction.confidence * sentiment_multiplier * trigger_multiplier,
                4,
            )
            for kpi, weight in weights.items()
        }
        return ConversationKPIImpact(
            conversation_id=prediction.conversation_id,
            intent=prediction.intent,
            affected_kpis=affected_kpis,
            owner=self.OWNERS.get(prediction.intent, "Customer Operations"),
            priority=self._priority(prediction, affected_kpis),
        )

    def deployment_kpi_summary(self, decision: ConversationMonitoringDecision) -> dict[str, str]:
        if decision.status == "healthy":
            return {"analytics_reliability": "green", "contact_center_attention": "normal"}
        if decision.status == "critical":
            return {"analytics_reliability": "red", "contact_center_attention": "executive_review"}
        return {"analytics_reliability": "amber", "contact_center_attention": "team_review"}

    def _sentiment_multiplier(
        self,
        conversation_id: str,
        sentiments: list[SentimentPrediction],
    ) -> float:
        conversation_sentiments = [
            sentiment for sentiment in sentiments if sentiment.conversation_id == conversation_id
        ]
        if any(sentiment.sentiment == "negative" for sentiment in conversation_sentiments):
            return 1.25
        if any(sentiment.sentiment == "positive" for sentiment in conversation_sentiments):
            return 0.8
        return 1.0

    def _priority(self, prediction: IntentPrediction, affected_kpis: dict[str, float]) -> str:
        max_impact = max(affected_kpis.values(), default=0.0)
        if "churn_risk" in prediction.triggers or max_impact >= 0.75:
            return "p1"
        if prediction.triggers or max_impact >= 0.45:
            return "p2"
        return "p3"
