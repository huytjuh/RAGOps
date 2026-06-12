from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TesseractOCRConfig:
    language: str = "eng"
    page_segmentation_mode: int = 6
    engine_mode: int = 3
    tesseract_cmd: str | None = None


@dataclass(frozen=True)
class OCRResult:
    source_path: str | None
    text: str
    pages: list[str]
    confidence: float | None
    engine: str
    metadata: dict[str, str] = field(default_factory=dict)


class TesseractOCRService:
    """Extract document text with Tesseract OCR, with text-file passthrough."""

    TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".html"}
    IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

    def __init__(self, config: TesseractOCRConfig | None = None) -> None:
        self.config = config or TesseractOCRConfig()

    def extract_text(
        self,
        source: str | Path | None = None,
        raw_text: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> OCRResult:
        if raw_text is not None:
            return OCRResult(
                source_path=str(source) if source else None,
                text=self._normalize(raw_text),
                pages=[self._normalize(raw_text)],
                confidence=None,
                engine="raw_text",
                metadata=metadata or {},
            )

        if source is None:
            raise ValueError("Either source or raw_text must be provided.")

        path = Path(source)
        if path.suffix.lower() in self.TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8")
            return OCRResult(
                source_path=str(path),
                text=self._normalize(text),
                pages=[self._normalize(text)],
                confidence=None,
                engine="text_passthrough",
                metadata=metadata or {},
            )

        if path.suffix.lower() not in self.IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported document type for OCR: {path.suffix}")

        return self._extract_image_text(path, metadata or {})

    def _extract_image_text(self, path: Path, metadata: dict[str, str]) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "Tesseract OCR requires pytesseract and Pillow for image documents."
            ) from error

        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

        config = f"--oem {self.config.engine_mode} --psm {self.config.page_segmentation_mode}"
        with Image.open(path) as image:
            text = pytesseract.image_to_string(image, lang=self.config.language, config=config)
            data = pytesseract.image_to_data(
                image,
                lang=self.config.language,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        confidences = [
            float(confidence)
            for confidence in data.get("conf", [])
            if str(confidence).replace(".", "", 1).lstrip("-").isdigit() and float(confidence) >= 0
        ]
        confidence = round(sum(confidences) / len(confidences), 4) if confidences else None
        normalized = self._normalize(text)
        return OCRResult(
            source_path=str(path),
            text=normalized,
            pages=[normalized],
            confidence=confidence,
            engine="tesseract",
            metadata=metadata,
        )

    def _normalize(self, text: str) -> str:
        lines = [" ".join(line.split()) for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()
