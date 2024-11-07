import os
import time
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog  # Поправленный импорт DatasetCatalog


def setup_config(dataset_dir, output_dir, max_iter=10000, batch_size=4, base_lr=0.001):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("deepfashion2_train",)
    cfg.DATASETS.TEST = ("deepfashion2_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = batch_size  # Размер батча
    cfg.SOLVER.BASE_LR = base_lr  # Скорость обучения
    cfg.SOLVER.MAX_ITER = max_iter  # Максимальное количество итераций (перезаписывается каждые 8 часов)
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Сохранение каждые 1000 итераций
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13  # Количество категорий одежды в DeepFashion2
    cfg.MODEL.WEIGHTS = "/home/mic0dy/models/R-50.pkl"
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def train_for_limited_time(cfg, max_seconds):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)  # Продолжение с последнего чекпоинта
    start_time = time.time()

    while time.time() - start_time < max_seconds:
        trainer.train()
        print("Training in progress...")

    print(f"Training stopped after {max_seconds / 60:.2f} minute(s). Progress saved.")


if __name__ == "__main__":
    # Настройки для вашего набора данных
    dataset_dir = "/mnt/d/Detect/DeepFashion2"  # Путь к вашим данным
    output_dir = "/mnt/d/Detect/COCO"  # Путь для сохранения модели и чекпоинтов

    # Регистрация датасета
    register_coco_instances("deepfashion2_train", {}, os.path.join(dataset_dir, "DeepFashion2_train_coco.json"),
                            os.path.join(dataset_dir, "train/image"))
    register_coco_instances("deepfashion2_val", {}, os.path.join(dataset_dir, "DeepFashion2_val_coco.json"),
                            os.path.join(dataset_dir, "validation/image"))

    # Проверка путей к файлам изображений для первых нескольких примеров из обучающей выборки
    print("Проверка путей к файлам изображений...")
    train_data = DatasetCatalog.get("deepfashion2_train")
    for idx, item in enumerate(train_data[:10]):  # Проверим первые 10 записей
        image_path = item.get('file_name', 'UNKNOWN')
        print(f"File Path: {image_path}")
        if not os.path.isfile(image_path):
            print(f"Warning: File does not exist at path: {image_path}")

    # Ввод времени обучения в минутах
    try:
        time_minutes = float(input("Введите количество минут для обучения (например, 30 для 30 минут): "))
        max_seconds = time_minutes * 60
    except ValueError:
        print("Пожалуйста, введите числовое значение для времени.")
        exit(1)

    # Настройка конфигурации
    cfg = setup_config(dataset_dir, output_dir)

    # Запуск обучения
    train_for_limited_time(cfg, max_seconds)
