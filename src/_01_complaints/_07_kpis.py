from __future__ import annotations

from dataclasses import dataclass

from _01_complaints._03_topic_classifier import ComplaintPrediction
from _01_complaints._06_monitoring import MonitoringDecision


@dataclass(frozen=True)
class BusinessKPIImpact:
    complaint_id: str
    label: str
    severity: str
    affected_kpis: dict[str, float]
    owner: str
    priority: str


class ComplaintKPIMapper:
    """Link complaint model outputs to operational business KPIs."""

    KPI_WEIGHTS = {
        "billing": {"refund_cost": 0.8, "first_contact_resolution": 0.5, "csat": 0.6},
        "service_quality": {"csat": 0.9, "agent_quality": 0.8, "churn_risk": 0.5},
        "technical_issue": {"digital_success_rate": 0.9, "repeat_contact_rate": 0.7, "csat": 0.5},
        "delivery_delay": {"sla_attainment": 0.9, "repeat_contact_rate": 0.6, "csat": 0.5},
        "compliance_risk": {"regulatory_exposure": 1.0, "case_escalation_rate": 0.9, "csat": 0.4},
    }
    OWNERS = {
        "billing": "Finance Operations",
        "service_quality": "Customer Care",
        "technical_issue": "Digital Product",
        "delivery_delay": "Fulfillment Operations",
        "compliance_risk": "Compliance",
    }

    def map_prediction(self, prediction: ComplaintPrediction) -> BusinessKPIImpact:
        weights = self.KPI_WEIGHTS.get(prediction.label, {"csat": 0.4})
        severity_multiplier = {
            "low": 0.5,
            "medium": 0.75,
            "high": 1.0,
            "critical": 1.25,
        }[prediction.severity]
        affected_kpis = {
            kpi: round(weight * prediction.confidence * severity_multiplier, 4)
            for kpi, weight in weights.items()
        }
        return BusinessKPIImpact(
            complaint_id=prediction.complaint_id,
            label=prediction.label,
            severity=prediction.severity,
            affected_kpis=affected_kpis,
            owner=self.OWNERS.get(prediction.label, "Customer Operations"),
            priority=self._priority(prediction, affected_kpis),
        )

    def map_many(self, predictions: list[ComplaintPrediction]) -> list[BusinessKPIImpact]:
        return [self.map_prediction(prediction) for prediction in predictions]

    def deployment_kpi_summary(self, decision: MonitoringDecision) -> dict[str, str]:
        if decision.status == "healthy":
            return {"model_reliability": "green", "operations_attention": "normal"}
        if decision.status == "critical":
            return {"model_reliability": "red", "operations_attention": "executive_review"}
        return {"model_reliability": "amber", "operations_attention": "team_review"}

    def _priority(self, prediction: ComplaintPrediction, affected_kpis: dict[str, float]) -> str:
        max_impact = max(affected_kpis.values(), default=0.0)
        if prediction.severity == "critical" or max_impact >= 0.75:
            return "p1"
        if prediction.severity == "high" or max_impact >= 0.45:
            return "p2"
        return "p3"
