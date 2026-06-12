from __future__ import annotations

from dataclasses import dataclass

from _03_documents._00_schema import FieldRisk
from _03_documents._06_universal_extraction import ExtractedDocument


@dataclass(frozen=True)
class FieldEvaluation:
    field_name: str
    expected_value: str | None
    predicted_value: str | None
    exact_match: bool
    normalized_match: bool
    confidence: float
    required: bool
    risk: FieldRisk
    extraction_method: str
    error_type: str | None
    severity: str


@dataclass(frozen=True)
class DocumentEvaluationReport:
    extraction_completeness: float
    field_accuracy: float
    critical_field_accuracy: float
    layout_quality: float
    entity_density: float
    pii_redaction_coverage: float
    topic_confidence: float
    ocr_confidence: float | None
    trigger_count: int
    field_evaluations: list[FieldEvaluation]
    operational_triggers: list[str]
    quality_score: float


class DocumentEvaluator:
    """Evaluate document extraction quality, safety, and business readiness."""

    def evaluate(
        self,
        document: ExtractedDocument,
        required_fields: list[str] | None = None,
        expected_fields: dict[str, str] | None = None,
    ) -> DocumentEvaluationReport:
        completeness = self.extraction_completeness(document, required_fields or [])
        field_evaluations = self.field_evaluations(document, expected_fields or {})
        field_accuracy = self.field_accuracy(field_evaluations)
        critical_field_accuracy = self.critical_field_accuracy(field_evaluations)
        layout_quality = self.layout_quality(document)
        entity_density = self.entity_density(document)
        redaction_coverage = self.pii_redaction_coverage(document)
        topic_confidence = document.topic.confidence
        ocr_confidence = self._normalized_ocr_confidence(document)
        operational_triggers = self.operational_triggers(document, field_evaluations)

        quality_components = [
            completeness * 0.2,
            field_accuracy * 0.25,
            critical_field_accuracy * 0.15,
            layout_quality * 0.1,
            redaction_coverage * 0.2,
            topic_confidence * 0.05,
            (ocr_confidence if ocr_confidence is not None else 0.8) * 0.1,
        ]
        return DocumentEvaluationReport(
            extraction_completeness=completeness,
            field_accuracy=field_accuracy,
            critical_field_accuracy=critical_field_accuracy,
            layout_quality=layout_quality,
            entity_density=entity_density,
            pii_redaction_coverage=redaction_coverage,
            topic_confidence=topic_confidence,
            ocr_confidence=ocr_confidence,
            trigger_count=len(operational_triggers),
            field_evaluations=field_evaluations,
            operational_triggers=operational_triggers,
            quality_score=round(sum(quality_components), 4),
        )

    def extraction_completeness(
        self,
        document: ExtractedDocument,
        required_fields: list[str],
    ) -> float:
        if not required_fields:
            fields = [field for field in document.field_extractions.values() if field.required]
            if not fields:
                return 1.0 if document.structured_data else 0.0
            present = sum(field.is_present for field in fields)
            return round(present / len(fields), 4)
        present = sum(
            bool(
                document.field_extractions.get(field)
                and document.field_extractions[field].is_present
            )
            for field in required_fields
        )
        return round(present / len(required_fields), 4)

    def field_evaluations(
        self,
        document: ExtractedDocument,
        expected_fields: dict[str, str],
    ) -> list[FieldEvaluation]:
        evaluations = []
        for field_name, extraction in document.field_extractions.items():
            expected = expected_fields.get(field_name)
            predicted = extraction.value
            exact_match = expected is not None and predicted == expected
            normalized_match = (
                expected is not None
                and predicted is not None
                and self._normalize(expected) == self._normalize(predicted)
            )
            error_type = self._field_error_type(extraction.validation_errors, expected, predicted)
            evaluations.append(
                FieldEvaluation(
                    field_name=field_name,
                    expected_value=expected,
                    predicted_value=predicted,
                    exact_match=exact_match,
                    normalized_match=normalized_match,
                    confidence=extraction.confidence,
                    required=extraction.required,
                    risk=extraction.risk,
                    extraction_method=extraction.extraction_method,
                    error_type=error_type,
                    severity=self._severity(extraction.risk, error_type),
                )
            )
        return evaluations

    def field_accuracy(self, field_evaluations: list[FieldEvaluation]) -> float:
        required = [field for field in field_evaluations if field.required]
        if not required:
            return 1.0
        passed = sum(field.error_type is None for field in required)
        return round(passed / len(required), 4)

    def critical_field_accuracy(self, field_evaluations: list[FieldEvaluation]) -> float:
        critical = [
            field
            for field in field_evaluations
            if field.risk in {FieldRisk.HIGH, FieldRisk.CRITICAL} and field.required
        ]
        if not critical:
            return 1.0
        passed = sum(field.error_type is None for field in critical)
        return round(passed / len(critical), 4)

    def operational_triggers(
        self,
        document: ExtractedDocument,
        field_evaluations: list[FieldEvaluation],
    ) -> list[str]:
        triggers = set(document.topic.triggers)
        for evaluation in field_evaluations:
            if evaluation.error_type == "required_field_missing":
                triggers.add("required_high_risk_field_missing")
            if evaluation.error_type == "field_confidence_below_threshold":
                triggers.add("low_confidence_critical_field")
            if evaluation.error_type in {"invalid_date_format", "invalid_money_format"}:
                triggers.add("schema_validation_failed")
            if evaluation.severity == "critical":
                triggers.add("manual_review_required")
        if document.pii_findings:
            triggers.add("pii_detected")
        return sorted(triggers)

    def layout_quality(self, document: ExtractedDocument) -> float:
        has_sections = bool(document.layout.sections)
        has_key_values = bool(document.layout.key_values)
        has_tables = bool(document.layout.tables)
        score = 0.4 if has_sections else 0.0
        score += 0.4 if has_key_values else 0.0
        score += 0.2 if has_tables else 0.0
        return round(score, 4)

    def entity_density(self, document: ExtractedDocument) -> float:
        token_count = max(len(document.ocr.text.split()), 1)
        return round(min(len(document.entities) / token_count, 1.0), 4)

    def pii_redaction_coverage(self, document: ExtractedDocument) -> float:
        if not document.pii_findings:
            return 1.0
        covered = sum(
            finding.replacement in document.redacted_text for finding in document.pii_findings
        )
        return round(covered / len(document.pii_findings), 4)

    def _normalized_ocr_confidence(self, document: ExtractedDocument) -> float | None:
        if document.ocr.confidence is None:
            return None
        if document.ocr.confidence > 1:
            return round(document.ocr.confidence / 100, 4)
        return round(document.ocr.confidence, 4)

    def _field_error_type(
        self,
        validation_errors: tuple[str, ...],
        expected: str | None,
        predicted: str | None,
    ) -> str | None:
        if validation_errors:
            return validation_errors[0]
        if expected is not None and predicted is None:
            return "missing_expected_field"
        if expected is not None and self._normalize(expected) != self._normalize(predicted or ""):
            return "field_value_mismatch"
        return None

    def _severity(self, risk: FieldRisk, error_type: str | None) -> str:
        if error_type is None:
            return "none"
        if risk == FieldRisk.CRITICAL:
            return "critical"
        if risk == FieldRisk.HIGH:
            return "high"
        if risk == FieldRisk.MEDIUM:
            return "medium"
        return "low"

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().replace(",", "").split())
