from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from _03_documents._00_schema import (
    DocumentSchema,
    DocumentSchemaRegistry,
    FieldEvidence,
    FieldExtraction,
    FieldValidator,
    normalize_field_name,
)
from _03_documents._01_ocr import OCRResult, TesseractOCRService
from _03_documents._02_layout import DocumentLayout, LayoutAnalyzer
from _03_documents._03_entity_recognition import DocumentEntity, DocumentEntityRecognizer
from _03_documents._04_pii import PIIFinding, PIIRedactor
from _03_documents._05_topic_classifier import DocumentTopicClassifier, DocumentTopicPrediction


@dataclass(frozen=True)
class ExtractedDocument:
    document_id: str
    ocr: OCRResult
    layout: DocumentLayout
    entities: list[DocumentEntity]
    pii_findings: list[PIIFinding]
    redacted_text: str
    topic: DocumentTopicPrediction
    schema: DocumentSchema
    field_extractions: dict[str, FieldExtraction]
    structured_data: dict[str, object]


class UniversalDataExtractionPipeline:
    """Universal document extraction architecture for OCR and structured data."""

    def __init__(
        self,
        ocr: TesseractOCRService | None = None,
        layout_analyzer: LayoutAnalyzer | None = None,
        entity_recognizer: DocumentEntityRecognizer | None = None,
        pii_redactor: PIIRedactor | None = None,
        topic_classifier: DocumentTopicClassifier | None = None,
        schema_registry: DocumentSchemaRegistry | None = None,
        field_validator: FieldValidator | None = None,
    ) -> None:
        self.ocr = ocr or TesseractOCRService()
        self.layout_analyzer = layout_analyzer or LayoutAnalyzer()
        self.entity_recognizer = entity_recognizer or DocumentEntityRecognizer()
        self.pii_redactor = pii_redactor or PIIRedactor()
        self.topic_classifier = topic_classifier or DocumentTopicClassifier()
        self.schema_registry = schema_registry or DocumentSchemaRegistry()
        self.field_validator = field_validator or FieldValidator()

    def extract(
        self,
        document_id: str,
        source: str | Path | None = None,
        raw_text: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ExtractedDocument:
        ocr_result = self.ocr.extract_text(source=source, raw_text=raw_text, metadata=metadata)
        layout = self.layout_analyzer.analyze(ocr_result.text)
        entities = self.entity_recognizer.extract(ocr_result.text)
        redacted_text, pii_findings = self.pii_redactor.redact(ocr_result.text)
        topic = self.topic_classifier.predict(redacted_text)
        schema = self.schema_registry.get(topic.label)
        field_extractions = self._extract_schema_fields(schema, layout, entities, ocr_result.text)
        structured_data = self._build_structured_data(
            layout,
            entities,
            pii_findings,
            topic,
            schema,
            field_extractions,
        )

        return ExtractedDocument(
            document_id=document_id,
            ocr=ocr_result,
            layout=layout,
            entities=entities,
            pii_findings=pii_findings,
            redacted_text=redacted_text,
            topic=topic,
            schema=schema,
            field_extractions=field_extractions,
            structured_data=structured_data,
        )

    def _extract_schema_fields(
        self,
        schema: DocumentSchema,
        layout: DocumentLayout,
        entities: list[DocumentEntity],
        text: str,
    ) -> dict[str, FieldExtraction]:
        normalized_key_values = {
            normalize_field_name(key): value for key, value in layout.key_values.items()
        }
        fields: dict[str, FieldExtraction] = {}

        for field_spec in schema.fields:
            candidate_names = (field_spec.name, *field_spec.aliases)
            value = None
            method = "missing"
            confidence = 0.0
            evidence = None

            for candidate_name in candidate_names:
                key = normalize_field_name(candidate_name)
                if normalized_key_values.get(key):
                    value = normalized_key_values[key]
                    confidence = self._confidence_from_source(field_spec.name, value, "key_value")
                    method = "key_value"
                    evidence = FieldEvidence(
                        source="layout.key_values",
                        text=f"{candidate_name}: {value}",
                    )
                    break

            if value is None:
                value, confidence, evidence = self._field_from_entities(field_spec.name, entities)
                if value is not None:
                    method = "entity"

            if value is not None and evidence is None:
                start = text.find(value)
                evidence = FieldEvidence(
                    source=method,
                    text=value,
                    page=1,
                    start=start if start >= 0 else None,
                    end=start + len(value) if start >= 0 else None,
                )

            validation_errors = self.field_validator.validate(field_spec, value, confidence)
            fields[field_spec.name] = FieldExtraction(
                field_name=field_spec.name,
                value=value,
                confidence=confidence,
                extraction_method=method,
                required=field_spec.required,
                risk=field_spec.risk,
                evidence=evidence,
                validation_errors=validation_errors,
            )
        return fields

    def _field_from_entities(
        self,
        field_name: str,
        entities: list[DocumentEntity],
    ) -> tuple[str | None, float, FieldEvidence | None]:
        entity_labels = {
            "amount_due": {"money"},
            "due_date": {"date"},
            "document_date": {"date"},
            "notice_date": {"date"},
            "effective_date": {"date"},
            "termination_date": {"date"},
            "response_deadline": {"date"},
            "invoice_number": {"identifier"},
            "case_number": {"identifier"},
            "reference_id": {"identifier"},
            "customer_name": {"person_or_org"},
            "party": {"person_or_org"},
            "regulator": {"person_or_org"},
        }
        labels = entity_labels.get(field_name, set())
        for entity in entities:
            if entity.label in labels:
                return (
                    entity.text,
                    entity.confidence,
                    FieldEvidence(
                        source=f"entity.{entity.label}",
                        text=entity.text,
                        page=1,
                        start=entity.start,
                        end=entity.end,
                    ),
                )
        return None, 0.0, None

    def _confidence_from_source(self, field_name: str, value: str, source: str) -> float:
        if not value:
            return 0.0
        if source == "key_value":
            return 0.9
        if field_name.endswith("_date") or field_name in {"amount_due", "invoice_number"}:
            return 0.85
        return 0.75

    def _build_structured_data(
        self,
        layout: DocumentLayout,
        entities: list[DocumentEntity],
        pii_findings: list[PIIFinding],
        topic: DocumentTopicPrediction,
        schema: DocumentSchema,
        field_extractions: dict[str, FieldExtraction],
    ) -> dict[str, object]:
        entity_groups: dict[str, list[str]] = {}
        for entity in entities:
            entity_groups.setdefault(entity.label, []).append(entity.text)

        return {
            "document_type": topic.label,
            "confidence": topic.confidence,
            "triggers": topic.triggers,
            "schema": schema.document_type,
            "fields": {
                name: {
                    "value": field.value,
                    "confidence": field.confidence,
                    "required": field.required,
                    "risk": field.risk.value,
                    "method": field.extraction_method,
                    "validation_errors": list(field.validation_errors),
                }
                for name, field in field_extractions.items()
            },
            "validation_errors": {
                name: list(field.validation_errors)
                for name, field in field_extractions.items()
                if field.validation_errors
            },
            "key_values": layout.key_values,
            "sections": [
                {"title": section.title, "text": section.text} for section in layout.sections
            ],
            "tables": [{"headers": table.headers, "rows": table.rows} for table in layout.tables],
            "entities": entity_groups,
            "pii_types": sorted({finding.pii_type for finding in pii_findings}),
        }
