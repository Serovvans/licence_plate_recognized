from pathlib import Path
from omegaconf import OmegaConf
import cv2

from src.ocr.trocr_recognizer import TrOCRRecognizer


def load_annotations(annotations_path: Path) -> dict:
    image_to_boxes = {}
    with open(annotations_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            row = line.strip().split(",")
            img_rel_path, x1, y1, x2, y2 = (
                row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4])
            )
            image_to_boxes.setdefault(img_rel_path, []).append((x1, y1, x2, y2))
    return image_to_boxes


def main():
    cfg = OmegaConf.load("config/inference/prediction.yaml")

    images_path = Path(cfg.images_path)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recognizer = TrOCRRecognizer(
        model_path=cfg.trocr_model_path,
        device=cfg.device,
    )
    image_to_boxes = load_annotations(Path(cfg.annotations_path))

    results = []
    for img_rel_path, boxes in image_to_boxes.items():
        image = cv2.imread(str(images_path / img_rel_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        for x1, y1, x2, y2 in boxes:
            pad = 4
            crop = image[
                max(0, y1 - pad):min(h, y2 + pad),
                max(0, x1 - pad):min(w, x2 + pad),
            ]
            if crop.size == 0:
                continue

            text = recognizer.recognize(crop)
            results.append(f"{img_rel_path},{text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("image_name,plate\n")
        f.write("\n".join(results))

    print(f"Готово. {len(results)} номеров → {output_path}")


if __name__ == "__main__":
    main()