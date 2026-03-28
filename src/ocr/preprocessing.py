import cv2
import numpy as np


def preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    if h < 64:
        scale = 64 / h
        crop = cv2.resize(crop, (int(w * scale), 64), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(2.0, (4, 4)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)