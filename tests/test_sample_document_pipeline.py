from _03_documents import (
    DocumentEvaluator,
    DocumentKPIMapper,
    DocumentMonitor,
    FieldRisk,
    UniversalDataExtractionPipeline,
)


def test_sample_document_pipeline_runs() -> None:
    text = """
    INVOICE
    Invoice Number: INV-1042
    Customer Name: Jane Carter
    Email: jane.carter@example.com
    Phone: 555-123-4567
    Amount Due: $1,240.00
    Due Date: 2026-07-01

    SERVICE SUMMARY
    Duplicate payment balance is overdue and requires refund review.
    | Item | Amount |
    | Subscription | $1,240.00 |
    """

    extracted = UniversalDataExtractionPipeline().extract(
        document_id="doc-001",
        raw_text=text,
        metadata={"source": "unit_test"},
    )

    assert extracted.topic.label == "invoice"
    assert extracted.layout.key_values["invoice_number"] == "INV-1042"
    assert extracted.pii_findings
    assert "[REDACTED_EMAIL]" in extracted.redacted_text
    assert "money" in {entity.label for entity in extracted.entities}
    assert extracted.structured_data["document_type"] == "invoice"
    assert extracted.field_extractions["amount_due"].risk == FieldRisk.CRITICAL
    assert extracted.field_extractions["amount_due"].is_valid
    assert extracted.structured_data["fields"]["amount_due"]["value"] == "$1,240.00"


def test_document_evaluation_monitoring_and_kpi_mapping_runs() -> None:
    text = """
    INVOICE
    Invoice Number: INV-1042
    Amount Due: $1,240.00
    Due Date: 2026-07-01
    Email: jane.carter@example.com
    Duplicate payment balance is overdue.
    """

    extracted = UniversalDataExtractionPipeline().extract("doc-002", raw_text=text)
    report = DocumentEvaluator().evaluate(
        extracted,
        required_fields=["invoice_number", "amount_due", "due_date"],
    )
    decision = DocumentMonitor().assess(report)
    impact = DocumentKPIMapper().map_document(extracted, decision)

    assert report.extraction_completeness == 1.0
    assert report.pii_redaction_coverage == 1.0
    assert decision.status in {"healthy", "warning", "critical"}
    assert impact.owner == "Finance Operations"
    assert impact.affected_kpis


def test_document_field_level_evaluation_detects_critical_failures() -> None:
    text = """
    INVOICE
    Invoice Number: INV-1042
    Customer Name: Jane Carter
    Amount Due: to be confirmed
    Due Date: tomorrow
    """

    extracted = UniversalDataExtractionPipeline().extract("doc-003", raw_text=text)
    report = DocumentEvaluator().evaluate(
        extracted,
        expected_fields={
            "invoice_number": "INV-1042",
            "amount_due": "$1,240.00",
            "due_date": "2026-07-01",
        },
    )
    decision = DocumentMonitor().assess(report)

    assert report.field_accuracy < 1.0
    assert report.critical_field_accuracy < 1.0
    assert "schema_validation_failed" in report.operational_triggers
    assert "manual_review_required" in report.operational_triggers
    assert decision.status == "critical"
