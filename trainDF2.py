import json
import os
from PIL import Image
import numpy as np
import cv2

def mask_to_polygons(mask):
    """ Преобразует бинарную маску в полигоны """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    return polygons

def deepfashion2_to_coco(dataset_dir, output_file, split):
    images = []
    annotations = []
    categories = [
        {"id": 1, "name": "short_sleeve_top"},
        {"id": 2, "name": "long_sleeve_top"},
        {"id": 3, "name": "short_sleeve_outwear"},
        {"id": 4, "name": "long_sleeve_outwear"},
        {"id": 5, "name": "vest"},
        {"id": 6, "name": "sling"},
        {"id": 7, "name": "shorts"},
        {"id": 8, "name": "trousers"},
        {"id": 9, "name": "skirt"},
        {"id": 10, "name": "short_sleeve_dress"},
        {"id": 11, "name": "long_sleeve_dress"},
        {"id": 12, "name": "vest_dress"},
        {"id": 13, "name": "sling_dress"},
    ]

    image_id = 0
    annotation_id = 0

    # Указываем путь к изображениям и аннотациям
    image_dir = os.path.join(dataset_dir, split, "image")
    anno_dir = os.path.join(dataset_dir, split, "annos")

    for file in os.listdir(image_dir):
        if file.endswith(".jpg"):
            image_id += 1
            img_path = os.path.join(image_dir, file)
            img = Image.open(img_path)
            width, height = img.size

            images.append({
                "id": image_id,
                "file_name": f"{split}/image/{file}",
                "width": width,
                "height": height
            })

            # Загрузка соответствующей аннотации
            annotation_path = os.path.join(anno_dir, file.replace(".jpg", ".json"))
            if os.path.exists(annotation_path):
                with open(annotation_path) as f:
                    annotation_data = json.load(f)

                for obj in annotation_data["objects"]:
                    category_id = obj["category_id"]
                    mask_path = os.path.join(anno_dir, obj["mask_path"])
                    mask = np.array(Image.open(mask_path).convert("L"))
                    mask[mask > 0] = 255

                    # Преобразуем маску в полигоны
                    segmentation = mask_to_polygons(mask)
                    if not segmentation:
                        continue

                    x, y, w, h = cv2.boundingRect(mask)

                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": segmentation,
                        "bbox": [x, y, w, h],
                        "iscrowd": 0,
                        "area": int(np.sum(mask > 0))
                    })
                    annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format, f)

# Генерация COCO-аннотаций для тренировочного и валидационного наборов
dataset_dir = "путь_к_DeepFashion2"  # Укажите путь к вашему набору данных
deepfashion2_to_coco(dataset_dir, "DeepFashion2_train_coco.json", "train")
deepfashion2_to_coco(dataset_dir, "DeepFashion2_val_coco.json", "validation")
print("COCO-аннотации сохранены.")
