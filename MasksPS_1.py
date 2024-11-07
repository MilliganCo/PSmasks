import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import subprocess
import os

# Загрузка модели YOLOv5 для сегментации
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model

# Изменение размеров изображения до кратных 32
def resize_to_multiple_of_32(image):
    h, w = image.shape[:2]
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    return cv2.resize(image, (new_w, new_h))

# Обработка изображения и создание маски одежды
def process_image(image_path, output_mask_path, model):
    # Загружаем изображение с помощью OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return

    # Изменяем размеры изображения до кратных 32
    image = resize_to_multiple_of_32(image)

    # Выполняем предсказание
    results = model(image)

    # Создаем пустую маску того же размера, что и изображение
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Получаем координаты объектов "person" и создаем маску
    for *box, conf, cls in results.xyxy[0].numpy():
        if int(cls) == 0:  # Проверка на класс "person"
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 255  # Заполняем белым цветом область объекта

    # Сохраняем маску как PNG изображение
    mask_img = Image.fromarray(mask)
    mask_img.save(output_mask_path)
    print(f"Маска одежды сохранена в {output_mask_path}")

# Окно для выбора путей к изображению и маске
def run_interface():
    root = tk.Tk()
    root.withdraw()

    # Выбор изображения
    image_path = filedialog.askopenfilename(title="Выберите изображение для обработки", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not image_path:
        print("Изображение не выбрано.")
        return

    # Выбор пути для сохранения маски
    output_mask_path = filedialog.asksaveasfilename(defaultextension=".png", title="Выберите место для сохранения маски", filetypes=[("PNG files", "*.png")])
    if not output_mask_path:
        print("Место для сохранения маски не выбрано.")
        return

    # Загружаем модель и обрабатываем изображение
    model = load_model()
    process_image(image_path, output_mask_path, model)

    # Запускаем Photoshop с изображением и маской
    open_in_photoshop(image_path, output_mask_path)

# Функция для открытия Photoshop с изображением и маской
def open_in_photoshop(image_path, mask_path):
    photoshop_path = r"D:\Adobe\Adobe Photoshop 2025\Photoshop.exe"  # Укажите здесь путь к Photoshop
    # Открываем Photoshop и загружаем исходное изображение и маску
    subprocess.run([photoshop_path, image_path, mask_path], shell=True)
    print("Photoshop открыт с изображением и маской.")

# Основная функция
def main():
    run_interface()

if __name__ == "__main__":
    main()
