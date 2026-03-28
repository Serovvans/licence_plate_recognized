from abc import ABC, abstractmethod
import numpy as np


class BaseRecognizer(ABC):
    @abstractmethod
    def recognize(self, crop: np.ndarray) -> str:
        pass