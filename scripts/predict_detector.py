from pathlib import Path
from omegaconf import OmegaConf
from ultralytics import YOLO


def main():
    cfg = OmegaConf.load("config/config.yaml")

    model = YOLO(cfg.inference.weights)

    image_paths = sorted(Path(cfg.inference.images_path).glob("*.jpg"))

    output_path = Path(cfg.inference.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("image_name,x_1,y_1,x_2,y_2,conf\n")

        for image_path in image_paths:
            results = model.predict(
                source=str(image_path),
                device=cfg.device,
                verbose=False,
            )

            image_name = f"test/{image_path.name}"

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = round(float(box.conf[0]), 4)
                    f.write(f"{image_name},{x1},{y1},{x2},{y2},{conf}\n")

    print(f"Готово. Предсказания сохранены в {output_path}")


if __name__ == "__main__":
    main()