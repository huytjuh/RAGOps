from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PIIFinding:
    text: str
    pii_type: str
    start: int
    end: int
    replacement: str


class PIIRedactor:
    """Detect and redact common PII from extracted document text."""

    PII_PATTERNS = {
        "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    }

    def detect(self, text: str) -> list[PIIFinding]:
        findings = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(
                    PIIFinding(
                        text=match.group(0),
                        pii_type=pii_type,
                        start=match.start(),
                        end=match.end(),
                        replacement=f"[REDACTED_{pii_type.upper()}]",
                    )
                )
        return sorted(findings, key=lambda finding: (finding.start, finding.end))

    def redact(self, text: str) -> tuple[str, list[PIIFinding]]:
        findings = self.detect(text)
        redacted = text
        for finding in reversed(findings):
            redacted = redacted[: finding.start] + finding.replacement + redacted[finding.end :]
        return redacted, findings
