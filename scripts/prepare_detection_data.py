from omegaconf import OmegaConf
from src.data_utils.yolo_utils import make_yolo_dataset


def main():
    cfg = OmegaConf.load("config/config.yaml")
    
    print("Создаём YOLO датасет...")
    path_to_config = make_yolo_dataset(cfg)
    print(f"Готово. Конфиг датасета: {path_to_config}")


if __name__ == "__main__":
    main()