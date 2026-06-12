from __future__ import annotations

from dataclasses import dataclass

from _03_documents._06_universal_extraction import ExtractedDocument
from _03_documents._07_monitoring import DocumentMonitoringDecision


@dataclass(frozen=True)
class DocumentKPIImpact:
    document_id: str
    document_type: str
    affected_kpis: dict[str, float]
    owner: str
    priority: str


class DocumentKPIMapper:
    """Link document processing outputs to business KPIs."""

    KPI_WEIGHTS = {
        "invoice": {
            "invoice_cycle_time": 0.8,
            "payment_accuracy": 0.8,
            "manual_review_rate": 0.5,
        },
        "contract": {
            "contract_cycle_time": 0.8,
            "obligation_capture_rate": 0.9,
            "legal_review_rate": 0.6,
        },
        "support_case": {
            "case_resolution_time": 0.8,
            "knowledge_capture_rate": 0.7,
            "repeat_contact_rate": 0.5,
        },
        "policy": {
            "policy_coverage": 0.8,
            "governance_readiness": 0.7,
            "audit_preparedness": 0.6,
        },
        "compliance_notice": {
            "regulatory_exposure": 1.0,
            "audit_risk": 0.9,
            "case_escalation_rate": 0.7,
        },
        "customer_correspondence": {
            "response_sla": 0.7,
            "customer_effort": 0.6,
            "csat": 0.5,
        },
    }
    OWNERS = {
        "invoice": "Finance Operations",
        "contract": "Legal Operations",
        "support_case": "Customer Support",
        "policy": "Governance",
        "compliance_notice": "Compliance",
        "customer_correspondence": "Customer Operations",
    }

    def map_document(
        self,
        document: ExtractedDocument,
        decision: DocumentMonitoringDecision | None = None,
    ) -> DocumentKPIImpact:
        weights = self.KPI_WEIGHTS.get(document.topic.label, {"manual_review_rate": 0.5})
        risk_multiplier = self._risk_multiplier(document, decision)
        affected_kpis = {
            kpi: round(weight * document.topic.confidence * risk_multiplier, 4)
            for kpi, weight in weights.items()
        }
        return DocumentKPIImpact(
            document_id=document.document_id,
            document_type=document.topic.label,
            affected_kpis=affected_kpis,
            owner=self.OWNERS.get(document.topic.label, "Operations"),
            priority=self._priority(document, decision, affected_kpis),
        )

    def deployment_kpi_summary(self, decision: DocumentMonitoringDecision) -> dict[str, str]:
        if decision.status == "healthy":
            return {"extraction_reliability": "green", "operations_attention": "normal"}
        if decision.status == "critical":
            return {"extraction_reliability": "red", "operations_attention": "manual_review"}
        return {"extraction_reliability": "amber", "operations_attention": "team_review"}

    def _risk_multiplier(
        self,
        document: ExtractedDocument,
        decision: DocumentMonitoringDecision | None,
    ) -> float:
        multiplier = 1.0 + (0.15 * len(document.topic.triggers))
        if document.pii_findings:
            multiplier += 0.1
        if decision and decision.status == "critical":
            multiplier += 0.3
        elif decision and decision.status == "warning":
            multiplier += 0.15
        return min(multiplier, 1.75)

    def _priority(
        self,
        document: ExtractedDocument,
        decision: DocumentMonitoringDecision | None,
        affected_kpis: dict[str, float],
    ) -> str:
        if decision and decision.status == "critical":
            return "p1"
        if "compliance_risk" in document.topic.triggers:
            return "p1"
        if max(affected_kpis.values(), default=0.0) >= 0.75:
            return "p2"
        return "p3"
