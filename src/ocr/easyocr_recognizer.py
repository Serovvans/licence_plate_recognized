import easyocr
import numpy as np
from src.ocr.base import BaseRecognizer
from src.ocr.postprocessing import postprocess


class EasyOCRRecognizer(BaseRecognizer):
    def __init__(self, languages: list[str], gpu: bool):
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
    def recognize(self, crop: np.ndarray) -> str:
        results = self.reader.readtext(crop, detail=1)
        if not results:
            return ""
        results = sorted(results, key=lambda x: x[0][0][1])
        raw_text = "".join(text for _, text, _ in results).strip()
        return postprocess(raw_text)