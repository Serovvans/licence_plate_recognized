from omegaconf import OmegaConf
from ultralytics import YOLO


def main():
    cfg = OmegaConf.load("config/config.yaml")

    model = YOLO(cfg.training.model)

    model.train(
        data=cfg.data.yolo_dataset_config,
        epochs=cfg.training.epochs,
        imgsz=cfg.data.img_size,
        batch=cfg.training.batch_size,
        device=cfg.device,
        seed=cfg.seed,
        project="runs/detect",
        name="hk_plates_v1",
        workers=cfg.training.workers,
        cache=cfg.training.cache,
    )


if __name__ == "__main__":
    main()