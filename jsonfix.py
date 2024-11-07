import json
import os

# Путь к JSON-файлу
json_file_path = "/mnt/d/Detect/DeepFashion2/DeepFashion2_train_coco.json"

# Откроем файл и исправим пути
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Обновляем пути
for image in data["images"]:
    if "train/image/train/image/" in image["file_name"]:
        image["file_name"] = image["file_name"].replace("train/image/train/image/", "train/image/")
    elif "train/image/image/" in image["file_name"]:
        image["file_name"] = image["file_name"].replace("train/image/image/", "train/image/")

# Сохраняем изменения
with open(json_file_path, 'w') as file:
    json.dump(data, file)

print("Paths in JSON have been updated successfully!")
