from pathlib import Path
from omegaconf import OmegaConf
import cv2

from src.ocr.factory import get_recognizer
from src.ocr.preprocessing import preprocess_plate_crop


def load_annotations(annotations_path: Path) -> dict:
    image_to_samples = {}
    with open(annotations_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            row = line.strip().split(",")
            # test формат: image_name,x1,y1,x2,y2 (без plate)
            img_rel_path, x1, y1, x2, y2 = (
                row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4])
            )
            image_to_samples.setdefault(img_rel_path, []).append((x1, y1, x2, y2))
    return image_to_samples


def main():
    cfg = OmegaConf.load("config/inference/prediction.yaml")

    images_path = Path(cfg.images_path)
    annotations_path = Path(cfg.annotations_path)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recognizer = get_recognizer(cfg)
    image_to_samples = load_annotations(annotations_path)

    results = []

    for img_rel_path, boxes in image_to_samples.items():
        img_full_path = images_path / img_rel_path
        image = cv2.imread(str(img_full_path))
        if image is None:
            print(f"Не удалось прочитать: {img_full_path}")
            continue

        h, w = image.shape[:2]

        for x1, y1, x2, y2 in boxes:
            pad = 4
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if cfg.preprocessing:
                crop = preprocess_plate_crop(crop)

            plate_text = recognizer.recognize(crop)
            results.append(f"{img_rel_path},{plate_text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("image_name,plate\n")
        f.write("\n".join(results))

    print(f"Готово. Результаты сохранены в {output_path} ({len(results)} номеров)")


if __name__ == "__main__":
    main()
    