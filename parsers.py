from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class DocumentElement:
    text: str
    element_type: str = "paragraph"
    page: int | None = None
    section_path: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser(Protocol):
    def parse(self, path: Path) -> list[DocumentElement]: ...


class MarkdownParser:
    _heading = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def parse(self, path: Path) -> list[DocumentElement]:
        text = path.read_text(encoding="utf-8")
        elements: list[DocumentElement] = []
        headings: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            content = "\n".join(buffer).strip()
            if content:
                elements.append(
                    DocumentElement(
                        text=content,
                        section_path=" > ".join(headings) or None,
                        element_type="markdown",
                    )
                )
            buffer.clear()

        for line in text.splitlines():
            match = self._heading.match(line)
            if match:
                flush()
                level = len(match.group(1))
                heading = match.group(2).strip()
                headings[:] = headings[: level - 1]
                headings.append(heading)
                elements.append(
                    DocumentElement(
                        text=heading,
                        section_path=" > ".join(headings),
                        element_type="heading",
                    )
                )
            else:
                buffer.append(line)
        flush()
        return elements


class PdfParser:
    def __init__(self, enable_ocr: bool | None = None) -> None:
        from app.core.config import get_settings

        settings = get_settings()
        self.enable_ocr = settings.ocr_enabled if enable_ocr is None else enable_ocr
        self.settings = settings

    def parse(self, path: Path) -> list[DocumentElement]:
        try:
            import fitz  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("解析 PDF 需要安装 pymupdf") from exc

        elements: list[DocumentElement] = []
        document = fitz.open(path)
        try:
            for page_number, page in enumerate(document, start=1):
                blocks = sorted(page.get_text("blocks"), key=lambda item: (item[1], item[0]))
                page_text_length = 0
                for block in blocks:
                    text = normalize_text(block[4])
                    if not text:
                        continue
                    page_text_length += len(text)
                    elements.append(
                        DocumentElement(
                            text=text,
                            page=page_number,
                            bbox=(
                                float(block[0]),
                                float(block[1]),
                                float(block[2]),
                                float(block[3]),
                            ),
                            element_type="pdf_text",
                        )
                    )

                has_embedded_images = bool(page.get_images(full=True))
                should_analyze_visuals = self.enable_ocr and (
                    page_text_length < self.settings.ocr_min_text_chars or has_embedded_images
                )
                if should_analyze_visuals:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    ocr_elements = ImageContentExtractor().extract(
                        pixmap.tobytes("png"),
                        source_type="pdf_page",
                        page_has_images=has_embedded_images,
                        run_ocr=page_text_length < self.settings.ocr_min_text_chars,
                    )
                    for element in ocr_elements:
                        element.page = page_number
                        element.metadata["ocr_fallback"] = True
                        element.metadata["pdf_embedded_images"] = has_embedded_images
                    elements.extend(ocr_elements)
        finally:
            document.close()
        return elements


class ImageParser:
    def parse(self, path: Path) -> list[DocumentElement]:
        return ImageContentExtractor().extract(path.read_bytes())


class TextParser:
    def parse(self, path: Path) -> list[DocumentElement]:
        return [DocumentElement(text=path.read_text(encoding="utf-8"), element_type="text")]


class PaddleOcrAdapter:
    """Small compatibility wrapper for PaddleOCR 2.x/3.x result shapes."""

    backend_name = "paddleocr"

    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "图片或扫描 PDF 需要 OCR 依赖，请执行 pip install -e '.[ocr]'"
            ) from exc
        self._ocr = PaddleOCR(use_doc_orientation_classify=True, lang="ch")

    def recognize_bytes(self, content: bytes) -> list[DocumentElement]:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("OCR 需要 pillow 和 numpy") from exc

        image = np.asarray(Image.open(io.BytesIO(content)).convert("RGB"))
        result = self._ocr.predict(image)
        elements: list[DocumentElement] = []
        for item in result or []:
            payload = getattr(item, "json", item)
            if callable(payload):
                payload = payload()
            if isinstance(payload, dict) and "res" in payload:
                payload = payload["res"]
            texts = payload.get("rec_texts", []) if isinstance(payload, dict) else []
            scores = payload.get("rec_scores", []) if isinstance(payload, dict) else []
            boxes = payload.get("rec_boxes", []) if isinstance(payload, dict) else []
            for index, text in enumerate(texts):
                clean = normalize_text(str(text))
                if not clean:
                    continue
                box = boxes[index] if index < len(boxes) else None
                bbox = (
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                    if box is not None
                    else None
                )
                score = float(scores[index]) if index < len(scores) else None
                elements.append(
                    DocumentElement(
                        text=clean,
                        element_type="ocr_text",
                        bbox=bbox,
                        confidence=score,
                    )
                )
        return elements


class RapidOcrAdapter:
    """ONNXRuntime OCR fallback, compatible with Python versions without Paddle wheels."""

    backend_name = "rapidocr_onnxruntime"

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import (  # type: ignore[import-not-found,import-untyped]
                RapidOCR,
            )
        except ImportError as exc:
            raise RuntimeError(
                "OCR 需要安装 rapidocr-onnxruntime，请执行 pip install -e '.[ocr]'"
            ) from exc
        self._ocr = RapidOCR()

    def recognize_bytes(self, content: bytes) -> list[DocumentElement]:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("OCR 需要 pillow 和 numpy") from exc

        image = np.asarray(Image.open(io.BytesIO(content)).convert("RGB"))
        result, _ = self._ocr(image)
        elements: list[DocumentElement] = []
        for item in result or []:
            if not item or len(item) < 3:
                continue
            box, text, score = item[0], str(item[1]), float(item[2])
            clean = normalize_text(text)
            if not clean:
                continue
            points = [point for point in box if len(point) >= 2]
            bbox = None
            if points:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                bbox = (min(xs), min(ys), max(xs), max(ys))
            elements.append(
                DocumentElement(
                    text=clean,
                    element_type="ocr_text",
                    bbox=bbox,
                    confidence=score,
                    metadata={"ocr_backend": self.backend_name},
                )
            )
        return elements


class ImageContentExtractor:
    def extract(
        self,
        content: bytes,
        source_type: str = "image",
        page_has_images: bool = False,
        run_ocr: bool = True,
    ) -> list[DocumentElement]:
        from app.core.config import get_settings

        settings = get_settings()
        elements: list[DocumentElement] = []
        ocr_error: RuntimeError | None = None
        if settings.ocr_enabled and run_ocr:
            elements, ocr_error = self._recognize(content, settings)

        text_length = sum(len(item.text) for item in elements)
        confidences = [item.confidence for item in elements if item.confidence is not None]
        average_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        weak_ocr = run_ocr and (
            text_length < settings.ocr_min_text_chars
            or bool(confidences and average_confidence < settings.ocr_min_confidence)
        )
        should_use_vlm = settings.vlm_enabled and (
            (source_type == "image" and settings.vlm_on_images)
            or (
                source_type == "pdf_page"
                and page_has_images
                and settings.vlm_on_pdf_pages_with_images
            )
            or weak_ocr
        )
        if should_use_vlm:
            from app.core.models import OllamaVisionClient

            try:
                description = normalize_text(OllamaVisionClient(settings).describe(content))
            except RuntimeError as exc:
                if not elements:
                    raise
                ocr_error = ocr_error or exc
            else:
                if description:
                    elements.append(
                        DocumentElement(
                            text=description,
                            element_type="vlm_description",
                            metadata={
                                "model": settings.ollama_vlm_model,
                                "source_type": source_type,
                            },
                        )
                    )
        if not elements and ocr_error:
            raise ocr_error
        return elements

    @staticmethod
    def _recognize(
        content: bytes,
        settings: Any,
    ) -> tuple[list[DocumentElement], RuntimeError | None]:
        adapters: list[type[PaddleOcrAdapter | RapidOcrAdapter]]
        if settings.ocr_backend == "paddle":
            adapters = [PaddleOcrAdapter]
        elif settings.ocr_backend == "rapidocr":
            adapters = [RapidOcrAdapter]
        else:
            adapters = [PaddleOcrAdapter, RapidOcrAdapter]
        errors: list[RuntimeError] = []
        for adapter_type in adapters:
            try:
                adapter = adapter_type()
                return adapter.recognize_bytes(content), None
            except (ImportError, RuntimeError) as exc:
                errors.append(exc if isinstance(exc, RuntimeError) else RuntimeError(str(exc)))
        return [], errors[-1] if errors else RuntimeError("OCR 未配置")


class ParserFactory:
    PDF_SUFFIXES = {".pdf"}
    MARKDOWN_SUFFIXES = {".md", ".markdown"}
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    TEXT_SUFFIXES = {".txt"}

    @classmethod
    def for_path(cls, path: Path) -> DocumentParser:
        suffix = path.suffix.lower()
        if suffix in cls.PDF_SUFFIXES:
            return PdfParser()
        if suffix in cls.MARKDOWN_SUFFIXES:
            return MarkdownParser()
        if suffix in cls.IMAGE_SUFFIXES:
            return ImageParser()
        if suffix in cls.TEXT_SUFFIXES:
            return TextParser()
        raise ValueError(f"不支持的文件格式：{suffix or '无扩展名'}")

    @classmethod
    def supported_suffixes(cls) -> set[str]:
        return cls.PDF_SUFFIXES | cls.MARKDOWN_SUFFIXES | cls.IMAGE_SUFFIXES | cls.TEXT_SUFFIXES


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()
