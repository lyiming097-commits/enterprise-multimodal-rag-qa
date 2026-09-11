from types import SimpleNamespace

from app.ingestion.parsers import DocumentElement, ImageContentExtractor


class FakeVisionClient:
    def __init__(self, settings: object) -> None:
        self.settings = settings

    def describe(self, image: bytes) -> str:
        assert image == b"image-bytes"
        return "图表显示企业收入逐年上升，2025 年为 100 万元。"


def settings(**overrides: object) -> SimpleNamespace:
    values = {
        "ocr_enabled": True,
        "ocr_backend": "auto",
        "ocr_min_text_chars": 30,
        "ocr_min_confidence": 0.75,
        "vlm_enabled": True,
        "vlm_on_images": True,
        "vlm_on_pdf_pages_with_images": True,
        "ollama_vlm_model": "qwen2.5vl:7b",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_image_combines_ocr_and_vlm(monkeypatch) -> None:
    extractor = ImageContentExtractor()
    monkeypatch.setattr(
        extractor,
        "_recognize",
        lambda content, config: (
            [
                DocumentElement(
                    text="企业收入表",
                    element_type="ocr_text",
                    confidence=0.96,
                    metadata={"ocr_backend": "rapidocr_onnxruntime"},
                )
            ],
            None,
        ),
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings())
    monkeypatch.setattr("app.core.models.OllamaVisionClient", FakeVisionClient)

    elements = extractor.extract(b"image-bytes")

    assert [item.element_type for item in elements] == ["ocr_text", "vlm_description"]
    assert elements[1].metadata == {"model": "qwen2.5vl:7b", "source_type": "image"}


def test_text_pdf_image_uses_vlm_without_duplicate_ocr(monkeypatch) -> None:
    extractor = ImageContentExtractor()
    monkeypatch.setattr(
        extractor,
        "_recognize",
        lambda content, config: (_ for _ in ()).throw(AssertionError("OCR should not run")),
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings())
    monkeypatch.setattr("app.core.models.OllamaVisionClient", FakeVisionClient)

    elements = extractor.extract(
        b"image-bytes",
        source_type="pdf_page",
        page_has_images=True,
        run_ocr=False,
    )

    assert len(elements) == 1
    assert elements[0].element_type == "vlm_description"
