from __future__ import annotations

from dataclasses import dataclass, field

from _03_documents._07_evaluation import DocumentEvaluationReport


@dataclass(frozen=True)
class DocumentMonitoringDecision:
    status: str
    alerts: list[str]
    recommended_actions: list[str]
    metrics: dict[str, float | int | None] = field(default_factory=dict)


class DocumentMonitor:
    """Monitor document extraction quality in deployment."""

    def __init__(
        self,
        min_quality_score: float = 0.75,
        min_extraction_completeness: float = 0.9,
        min_field_accuracy: float = 0.9,
        min_critical_field_accuracy: float = 1.0,
        min_layout_quality: float = 0.6,
        min_topic_confidence: float = 0.6,
        min_pii_redaction_coverage: float = 1.0,
        min_ocr_confidence: float = 0.7,
        max_trigger_count: int = 2,
    ) -> None:
        self.min_quality_score = min_quality_score
        self.min_extraction_completeness = min_extraction_completeness
        self.min_field_accuracy = min_field_accuracy
        self.min_critical_field_accuracy = min_critical_field_accuracy
        self.min_layout_quality = min_layout_quality
        self.min_topic_confidence = min_topic_confidence
        self.min_pii_redaction_coverage = min_pii_redaction_coverage
        self.min_ocr_confidence = min_ocr_confidence
        self.max_trigger_count = max_trigger_count

    def assess(self, report: DocumentEvaluationReport) -> DocumentMonitoringDecision:
        alerts: list[str] = []
        actions: list[str] = []

        if report.quality_score < self.min_quality_score:
            alerts.append("document_quality_below_threshold")
            actions.append("Review OCR, layout extraction, and field mapping before automation.")
        if report.extraction_completeness < self.min_extraction_completeness:
            alerts.append("required_fields_missing")
            actions.append("Route document to manual validation or improve extraction templates.")
        if report.field_accuracy < self.min_field_accuracy:
            alerts.append("field_accuracy_below_threshold")
            actions.append("Review schema mapping and collect corrected labels for failed fields.")
        if report.critical_field_accuracy < self.min_critical_field_accuracy:
            alerts.append("critical_field_accuracy_below_threshold")
            actions.append("Block straight-through processing and require human validation.")
        if report.layout_quality < self.min_layout_quality:
            alerts.append("layout_quality_below_threshold")
            actions.append("Tune section, table, and key-value parsing for this document type.")
        if report.topic_confidence < self.min_topic_confidence:
            alerts.append("document_type_confidence_below_threshold")
            actions.append("Review document taxonomy and classifier examples.")
        if report.pii_redaction_coverage < self.min_pii_redaction_coverage:
            alerts.append("pii_redaction_incomplete")
            actions.append("Block downstream use until PII redaction is corrected.")
        if report.ocr_confidence is not None and report.ocr_confidence < self.min_ocr_confidence:
            alerts.append("ocr_confidence_below_threshold")
            actions.append("Improve scan quality or route to human review.")
        if report.trigger_count > self.max_trigger_count:
            alerts.append("document_trigger_volume_above_threshold")
            actions.append("Escalate compliance, payment, or contractual risk triggers.")
        if "manual_review_required" in report.operational_triggers:
            alerts.append("manual_review_required")
            actions.append("Send the document and field evidence to the validation queue.")

        status = "healthy" if not alerts else "warning"
        critical_alerts = {
            "pii_redaction_incomplete",
            "required_fields_missing",
            "critical_field_accuracy_below_threshold",
            "manual_review_required",
        }
        if critical_alerts.intersection(alerts):
            status = "critical"

        return DocumentMonitoringDecision(
            status=status,
            alerts=alerts,
            recommended_actions=actions,
            metrics={
                "extraction_completeness": report.extraction_completeness,
                "field_accuracy": report.field_accuracy,
                "critical_field_accuracy": report.critical_field_accuracy,
                "layout_quality": report.layout_quality,
                "entity_density": report.entity_density,
                "pii_redaction_coverage": report.pii_redaction_coverage,
                "topic_confidence": report.topic_confidence,
                "ocr_confidence": report.ocr_confidence,
                "trigger_count": report.trigger_count,
                "quality_score": report.quality_score,
            },
        )
