install-conda:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh

setup:
	conda env create -f environment.yml
	conda run -n kc-plates pip install -e .

update-env:
	conda env update -f environment.yml --prune

cuda-install:
	pip install "torch==2.4.0" torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
	pip install "numpy<2" --force-reinstall

download:
	python scripts/download_data.py

dataset_detection:
	python scripts/prepare_detection_data.py

train-detector:
	python scripts/train_detector.py

predict:
	python scripts/predict_detector.py

predict-ocr:
	python scripts/predict_ocr.py

data-trocr:
	python scripts/prepare_trocr_data.py

train-trocr:
	python scripts/train_trocr.py

predict-trocr:
	python scripts/predict_trocr.py