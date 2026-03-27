setup:
	conda env create -f environment.yml
	conda run -n kc-plates pip install -e .

update-env:
	conda env update -f environment.yml --prune

dataset_detection:
	python scripts/prepare_detection_data.py

train-detector:
	python scripts/train_detector.py

predict:
	python scripts/predict_detector.py