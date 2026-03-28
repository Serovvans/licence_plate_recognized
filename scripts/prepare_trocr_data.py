import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf


def main():
    cfg = OmegaConf.load("config/training/trocr.yaml")

    output_dir = Path(cfg.data.processed_output_path) / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    root_img_path = Path(cfg.data.raw_images_path)
    metadata = []

    with open(cfg.data.raw_annotations_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]

    for i, line in enumerate(tqdm(lines, desc="Preparing crops")):
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue

        img_rel_path = parts[0]
        x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        text = parts[5].strip()

        image = cv2.imread(str(root_img_path / img_rel_path))
        if image is None:
            continue

        h_img, w_img = image.shape[:2]
        pad = 3
        crop = image[
            max(0, y1 - pad):min(h_img, y2 + pad),
            max(0, x1 - pad):min(w_img, x2 + pad),
        ]
        if crop.size == 0:
            continue

        file_name = f"crop_{i}.jpg"
        cv2.imwrite(str(output_dir / file_name), crop)
        metadata.append({"file_name": file_name, "text": text})

    pd.DataFrame(metadata).to_csv(output_dir / "metadata.csv", index=False)
    print(f"Готово. Кропов: {len(metadata)}")


if __name__ == "__main__":
    main()