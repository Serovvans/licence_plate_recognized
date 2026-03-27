import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def draw_result(image_path: str, annotation_path: str) -> None:
    """Отрисовывает боксы и подписи номеров по аннотациями в формате изначального датасета

    Args:
        image_paths (str): директория с изображениями
        annotations_path (str): _description_
    """
    image_paths = Path(image_path).rglob(f"train/*.jpg")
    annotations_path = Path(annotation_path)
    
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = []
        f.readline()
        for ann in f.readlines():
            row = ann.split(",")
            annotations.append(
                [row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4]), row[5].strip()]
            )
        
    n = 5
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n))

    for i, path in enumerate(image_paths):
        if i >= n:
            break
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_id = f"{path.parent.name}/{path.name}"
        image_boxes = [annotation for annotation in annotations if annotation[0] == image_id]
        
        for box in image_boxes:
            x1 = box[1]
            y1 = box[2]
            x2 = box[3]
            y2 = box[4]

            color = tuple(map(int, np.random.randint(0, 255, 3)))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
            cv2.putText(image,'%s'%(box[5]), (x1, y1-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)
        axes[i].imshow(image)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    