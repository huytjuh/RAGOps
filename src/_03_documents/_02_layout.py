from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentSection:
    title: str
    text: str
    order: int


@dataclass(frozen=True)
class DocumentTable:
    headers: list[str]
    rows: list[dict[str, str]]
    order: int


@dataclass(frozen=True)
class DocumentLayout:
    sections: list[DocumentSection]
    tables: list[DocumentTable]
    key_values: dict[str, str] = field(default_factory=dict)


class LayoutAnalyzer:
    """Extract sections, simple tables, and key-value fields from OCR text."""

    HEADING_PATTERN = re.compile(r"^[A-Z][A-Z0-9 \-/&]{2,}$")
    KEY_VALUE_PATTERN = re.compile(r"^\s*([^:]{2,50})\s*:\s*(.+?)\s*$")

    def analyze(self, text: str) -> DocumentLayout:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections = self._extract_sections(lines)
        tables = self._extract_tables(lines)
        key_values = self._extract_key_values(lines)
        return DocumentLayout(sections=sections, tables=tables, key_values=key_values)

    def _extract_sections(self, lines: list[str]) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        current_title = "document"
        current_lines: list[str] = []

        def flush() -> None:
            if current_lines:
                sections.append(
                    DocumentSection(
                        title=current_title.lower().replace(" ", "_"),
                        text=" ".join(current_lines),
                        order=len(sections),
                    )
                )

        for line in lines:
            if self._is_heading(line):
                flush()
                current_title = line
                current_lines = []
            else:
                current_lines.append(line)
        flush()

        if not sections and lines:
            sections.append(DocumentSection(title="document", text=" ".join(lines), order=0))
        return sections

    def _extract_tables(self, lines: list[str]) -> list[DocumentTable]:
        tables: list[DocumentTable] = []
        table_lines = [line for line in lines if "|" in line]
        if len(table_lines) < 2:
            return tables

        headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
        rows = []
        for line in table_lines[1:]:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != len(headers):
                continue
            rows.append(dict(zip(headers, cells, strict=True)))
        if rows:
            tables.append(DocumentTable(headers=headers, rows=rows, order=0))
        return tables

    def _extract_key_values(self, lines: list[str]) -> dict[str, str]:
        key_values = {}
        for line in lines:
            match = self.KEY_VALUE_PATTERN.match(line)
            if match:
                key = match.group(1).strip().lower().replace(" ", "_")
                key_values[key] = match.group(2).strip()
        return key_values

    def _is_heading(self, line: str) -> bool:
        if len(line.split()) > 8:
            return False
        return bool(self.HEADING_PATTERN.match(line)) or line.endswith(":") and len(line) < 50
