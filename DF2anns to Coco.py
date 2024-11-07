import json
import os
from PIL import Image
import numpy as np
import cv2

def mask_to_polygons(mask):
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
        {"id": 1, "name": "short sleeve top"},
        {"id": 2, "name": "long sleeve top"},
        {"id": 3, "name": "short sleeve outwear"},
        {"id": 4, "name": "long sleeve outwear"},
        {"id": 5, "name": "vest"},
        {"id": 6, "name": "sling"},
        {"id": 7, "name": "shorts"},
        {"id": 8, "name": "trousers"},
        {"id": 9, "name": "skirt"},
        {"id": 10, "name": "short sleeve dress"},
        {"id": 11, "name": "long sleeve dress"},
        {"id": 12, "name": "vest dress"},
        {"id": 13, "name": "sling dress"},
    ]

    image_id = 0
    annotation_id = 0

    image_dir = os.path.join(dataset_dir, split, "image")
    anno_dir = os.path.join(dataset_dir, split, "annos")

    for file in os.listdir(image_dir):
        if file.endswith(".jpg"):
            image_id += 1
            img_path = os.path.join(image_dir, file)
            print(f"Processing image: {img_path}")  # Сообщение о текущем изображении
            img = Image.open(img_path)
            width, height = img.size

            images.append({
                "id": image_id,
                "file_name": f"{split}/image/{file}",
                "width": width,
                "height": height
            })

            annotation_path = os.path.join(anno_dir, file.replace(".jpg", ".json"))
            if os.path.exists(annotation_path):
                with open(annotation_path) as f:
                    annotation_data = json.load(f)

                item_index = 1
                while f"item{item_index}" in annotation_data:
                    obj = annotation_data[f"item{item_index}"]
                    category_id = obj["category_id"]
                    bbox = obj["bounding_box"]
                    segmentation = obj["segmentation"]

                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": segmentation,
                        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                        "iscrowd": 0,
                        "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    })
                    annotation_id += 1
                    item_index += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    print(f"Saving COCO annotations to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(coco_format, f)
    print("Conversion completed.")

# Пример использования
dataset_dir = "/mnt/d/Detect/DeepFashion2"
output_dir = "/mnt/d/Detect/COCO"
os.makedirs(output_dir, exist_ok=True)

deepfashion2_to_coco(dataset_dir, os.path.join(output_dir, "DeepFashion2_train_coco.json"), "train")
deepfashion2_to_coco(dataset_dir, os.path.join(output_dir, "DeepFashion2_val_coco.json"), "validation")
