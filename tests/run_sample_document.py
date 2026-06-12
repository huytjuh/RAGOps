from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from _03_documents import (  # noqa: E402
    DocumentEvaluator,
    DocumentKPIMapper,
    DocumentMonitor,
    UniversalDataExtractionPipeline,
)


def main() -> None:
    sample_document = """
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
        raw_text=sample_document,
        metadata={"source": "sample_runner"},
    )
    report = DocumentEvaluator().evaluate(
        extracted,
        required_fields=["invoice_number", "amount_due", "due_date"],
    )
    decision = DocumentMonitor().assess(report)
    kpi_mapper = DocumentKPIMapper()
    impact = kpi_mapper.map_document(extracted, decision)

    output = {
        "extracted_document": asdict(extracted),
        "evaluation": asdict(report),
        "monitoring": asdict(decision),
        "business_kpi_impact": asdict(impact),
        "deployment_kpi_summary": kpi_mapper.deployment_kpi_summary(decision),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
