from omegaconf import DictConfig
from src.ocr.base import BaseRecognizer
from src.ocr.easyocr_recognizer import EasyOCRRecognizer


def get_recognizer(cfg: DictConfig) -> BaseRecognizer:
    if cfg.recognizer == "easyocr":
        return EasyOCRRecognizer(
            languages=list(cfg.languages),
            gpu=cfg.gpu,
        )
    # Сюда потом добавишь:
    if cfg.recognizer == "trocr":
        from src.ocr.trocr_recognizer import TrOCRRecognizer
        return TrOCRRecognizer(
            model_path=cfg.trocr_model_path,
            device=cfg.device,
        )
    raise ValueError(f"Неизвестный recognizer: {cfg.recognizer}")