from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float
    source: str


class DocumentEntityRecognizer:
    """Extract common entities from documents using patterns and optional NLP hooks."""

    ENTITY_PATTERNS = {
        "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
        "money": re.compile(r"(?:\b(?:USD|EUR|GBP)\s?|\$|€|£)\d+(?:,\d{3})*(?:\.\d{2})?\b"),
        "date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
        "identifier": re.compile(
            r"\b(?:invoice|case|account|order)[ _-]?(?:id|number|no)?[: #]*([A-Z0-9-]{4,})\b",
            re.I,
        ),
    }

    def extract(self, text: str) -> list[DocumentEntity]:
        entities = []
        for label, pattern in self.ENTITY_PATTERNS.items():
            for match in pattern.finditer(text):
                entity_text = (
                    match.group(1) if label == "identifier" and match.groups() else match.group(0)
                )
                entities.append(
                    DocumentEntity(
                        text=entity_text,
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                        source="regex",
                    )
                )
        entities.extend(self._capitalized_name_candidates(text))
        return sorted(entities, key=lambda entity: (entity.start, entity.end))

    def _capitalized_name_candidates(self, text: str) -> list[DocumentEntity]:
        pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
        blocked = {"Customer Name", "Invoice Number", "Due Date", "Service Summary"}
        entities = []
        for match in pattern.finditer(text):
            value = match.group(1)
            if value in blocked:
                continue
            entities.append(
                DocumentEntity(
                    text=value,
                    label="person_or_org",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.55,
                    source="rule",
                )
            )
        return entities
