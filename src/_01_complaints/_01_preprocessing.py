from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProcessedComplaint:
    """Normalized complaint text plus traceable metadata."""

    complaint_id: str
    raw_text: str
    text: str
    tokens: list[str]
    metadata: dict[str, str] = field(default_factory=dict)


class ComplaintPreprocessor:
    """Prepare complaint text for BERT classification and topic discovery."""

    def __init__(self, min_token_length: int = 2) -> None:
        self.min_token_length = min_token_length

    def transform(
        self,
        complaint_id: str,
        text: str,
        metadata: dict[str, str] | None = None,
    ) -> ProcessedComplaint:
        normalized = self.normalize(text)
        tokens = self.tokenize(normalized)
        return ProcessedComplaint(
            complaint_id=complaint_id,
            raw_text=text,
            text=normalized,
            tokens=tokens,
            metadata=metadata or {},
        )

    def transform_many(self, complaints: list[dict[str, str]]) -> list[ProcessedComplaint]:
        return [
            self.transform(
                complaint_id=complaint.get("complaint_id", str(index)),
                text=complaint["text"],
                metadata={k: v for k, v in complaint.items() if k not in {"complaint_id", "text"}},
            )
            for index, complaint in enumerate(complaints)
        ]

    def normalize(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " email_address ", text)
        text = re.sub(r"\b\d{4,}\b", " number ", text)
        text = re.sub(r"[^a-z0-9_\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> list[str]:
        return [token for token in text.split() if len(token) >= self.min_token_length]
