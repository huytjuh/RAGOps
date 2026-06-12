from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum


class FieldRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    required: bool = True
    risk: FieldRisk = FieldRisk.MEDIUM
    value_type: str = "text"
    aliases: tuple[str, ...] = ()
    min_confidence: float = 0.8
    allowed_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentSchema:
    document_type: str
    fields: tuple[FieldSpec, ...]
    automation_min_confidence: float = 0.85

    @property
    def required_fields(self) -> tuple[str, ...]:
        return tuple(field_spec.name for field_spec in self.fields if field_spec.required)


@dataclass(frozen=True)
class FieldEvidence:
    source: str
    text: str
    page: int | None = None
    start: int | None = None
    end: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class FieldExtraction:
    field_name: str
    value: str | None
    confidence: float
    extraction_method: str
    required: bool
    risk: FieldRisk
    evidence: FieldEvidence | None = None
    validation_errors: tuple[str, ...] = ()

    @property
    def is_present(self) -> bool:
        return bool(self.value)

    @property
    def is_valid(self) -> bool:
        return self.is_present and not self.validation_errors


class DocumentSchemaRegistry:
    """Document-type field definitions used for governed extraction and evaluation."""

    def __init__(self, schemas: dict[str, DocumentSchema] | None = None) -> None:
        self.schemas = schemas or self._default_schemas()

    def get(self, document_type: str) -> DocumentSchema:
        return self.schemas.get(document_type, self.schemas["generic"])

    def _default_schemas(self) -> dict[str, DocumentSchema]:
        return {
            "invoice": DocumentSchema(
                document_type="invoice",
                fields=(
                    FieldSpec(
                        "invoice_number",
                        risk=FieldRisk.HIGH,
                        aliases=("invoice_no", "invoice_id"),
                        allowed_patterns=(r"^[A-Z0-9][A-Z0-9-]{3,}$",),
                    ),
                    FieldSpec(
                        "customer_name",
                        risk=FieldRisk.MEDIUM,
                        aliases=("customer", "client_name"),
                        min_confidence=0.75,
                    ),
                    FieldSpec(
                        "amount_due",
                        risk=FieldRisk.CRITICAL,
                        value_type="money",
                        aliases=("balance_due", "amount"),
                    ),
                    FieldSpec(
                        "due_date",
                        risk=FieldRisk.HIGH,
                        value_type="date",
                        aliases=("payment_due_date",),
                    ),
                ),
                automation_min_confidence=0.9,
            ),
            "contract": DocumentSchema(
                document_type="contract",
                fields=(
                    FieldSpec(
                        "party",
                        risk=FieldRisk.HIGH,
                        aliases=("counterparty", "customer_name"),
                    ),
                    FieldSpec("effective_date", risk=FieldRisk.HIGH, value_type="date"),
                    FieldSpec("signature", risk=FieldRisk.CRITICAL, min_confidence=0.9),
                    FieldSpec(
                        "termination_date",
                        required=False,
                        risk=FieldRisk.MEDIUM,
                        value_type="date",
                    ),
                ),
            ),
            "compliance_notice": DocumentSchema(
                document_type="compliance_notice",
                fields=(
                    FieldSpec("notice_date", risk=FieldRisk.HIGH, value_type="date"),
                    FieldSpec("regulator", risk=FieldRisk.HIGH),
                    FieldSpec(
                        "case_number",
                        risk=FieldRisk.CRITICAL,
                        aliases=("case_id", "case_no"),
                    ),
                    FieldSpec("response_deadline", risk=FieldRisk.CRITICAL, value_type="date"),
                ),
                automation_min_confidence=0.95,
            ),
            "generic": DocumentSchema(
                document_type="generic",
                fields=(
                    FieldSpec(
                        "document_date",
                        required=False,
                        risk=FieldRisk.MEDIUM,
                        value_type="date",
                    ),
                    FieldSpec("customer_name", required=False, risk=FieldRisk.MEDIUM),
                    FieldSpec("reference_id", required=False, risk=FieldRisk.MEDIUM),
                ),
                automation_min_confidence=0.8,
            ),
        }


class FieldValidator:
    """Validate extracted fields against schema-specific rules."""

    DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}/\d{1,2}/\d{2,4}$")
    MONEY_PATTERN = re.compile(
        r"^(?:USD|EUR|GBP)?\s?[$\u20ac\u00a3]?\s?\d+(?:,\d{3})*(?:\.\d{2})?$"
    )

    def validate(
        self,
        field_spec: FieldSpec,
        value: str | None,
        confidence: float,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        if field_spec.required and not value:
            errors.append("required_field_missing")
            return tuple(errors)
        if not value:
            return tuple(errors)
        if confidence < field_spec.min_confidence:
            errors.append("field_confidence_below_threshold")
        if field_spec.value_type == "date" and not self.DATE_PATTERN.match(value):
            errors.append("invalid_date_format")
        if field_spec.value_type == "money" and not self._is_money(value):
            errors.append("invalid_money_format")
        for pattern in field_spec.allowed_patterns:
            if not re.match(pattern, value):
                errors.append("field_pattern_mismatch")
                break
        return tuple(errors)

    def _is_money(self, value: str) -> bool:
        normalized = value.strip().replace(",", "")
        if self.MONEY_PATTERN.match(value.strip()):
            return True
        try:
            Decimal(normalized)
        except InvalidOperation:
            return False
        return True


def normalize_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
