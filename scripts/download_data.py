import zipfile
from pathlib import Path
from omegaconf import OmegaConf
import gdown


def main():
    cfg = OmegaConf.load("config/config.yaml")

    raw_path = Path(cfg.data.raw_path)
    raw_path.mkdir(parents=True, exist_ok=True)

    zip_path = raw_path / "data.zip"

    print("Скачиваем данные с Google Drive...")
    gdown.download(cfg.data.gdrive_url, str(zip_path), fuzzy=True)

    print("Распаковываем...")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(raw_path)

    zip_path.unlink()
    print(f"Готово. Данные в {raw_path}")


if __name__ == "__main__":
    main()