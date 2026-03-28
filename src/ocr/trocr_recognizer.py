import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.ocr.base import BaseRecognizer
from src.ocr.postprocessing import postprocess


class TrOCRRecognizer(BaseRecognizer):
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def recognize(self, crop: np.ndarray) -> str:
        image = Image.fromarray(crop[..., ::-1]).convert("RGB")  # BGR → RGB
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return postprocess(text)