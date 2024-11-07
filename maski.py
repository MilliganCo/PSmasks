import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from PIL import Image
import subprocess

# Загрузка конфигурации Detectron2 для сегментации
def load_detectron2_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Порог предсказания
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Использование GPU, если доступно
    return DefaultPredictor(cfg)

# Обработка изображения и создание маски одежды
def process_image_with_detectron2(image_path, output_mask_path, predictor):
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return

    # Выполняем предсказание
    outputs = predictor(image)
    instances = outputs["instances"]

    # Выводим классы всех предсказанных объектов, чтобы увидеть их индексы
    print("Классы предсказанных объектов:", instances.pred_classes)

    # Определяем классы, соответствующие одежде (в COCO dataset: 17=сумка, 18=галстук, 19=чемодан; попробуем для теста)
    classes_of_interest = [0]  # 0 = "person" для проверки, замените на фактические индексы одежды после проверки

    # Получаем маски для классов одежды
    mask_combined = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    for i in range(len(instances)):
        if instances.pred_classes[i].item() in classes_of_interest:
            mask = instances.pred_masks[i].cpu().numpy()
            mask_combined[mask] = 255  # Объединяем маски одежды

    # Сохраняем маску как PNG
    mask_img = Image.fromarray(mask_combined)
    mask_img.save(output_mask_path)
    print(f"Маска одежды сохранена в {output_mask_path}")

# Функция для открытия изображения и маски в Photoshop
# Функция для открытия изображения и маски в Photoshop
def open_in_photoshop(image_path, mask_path):
    photoshop_path = "/mnt/d/Adobe/Adobe Photoshop 2025/Photoshop.exe"  # Путь к Photoshop без кавычек
    subprocess.Popen([photoshop_path, image_path, mask_path])
    print("Photoshop открыт с изображением и маской.")


# Основная функция
def main():
    # Укажите пути к файлам, расположенным на Windows
    image_path = "/mnt/d/Detect/loc1241037395.jpg"  # Путь к вашему изображению на Windows
    output_mask_path = "/mnt/d/Detect/mask.png"  # Путь для сохранения маски

    # Загружаем модель и обрабатываем изображение
    predictor = load_detectron2_model()
    process_image_with_detectron2(image_path, output_mask_path, predictor)

    # Открываем Photoshop с изображением и маской
    open_in_photoshop(image_path, output_mask_path)

if __name__ == "__main__":
    main()
