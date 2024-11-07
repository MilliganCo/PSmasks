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

    # Получаем координаты предсказанных объектов
    predictions = results.xyxy[0].numpy()  # Получаем предсказания в формате numpy

    # Создаем пустую маску того же размера, что и изображение
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Предположим, что класс "person" для одежды - 0
    for *box, conf, cls in predictions:
        if int(cls) == 0:  # Проверяем, является ли объект человеком
            x1, y1, x2, y2 = map(int, box)  # Преобразуем координаты в int
            mask[y1:y2, x1:x2] = 255  # Заполняем область маски белым цветом

    # Сохраняем маску как PNG изображение
    mask_img = Image.fromarray(mask)
    mask_img.save(output_mask_path)
    print(f"Маска одежды сохранена в {output_mask_path}")


# Окно для выбора путей к изображению и маске
def run_interface():
    root = tk.Tk()
    root.withdraw()

    # Выбор изображения
    image_path = filedialog.askopenfilename(title="Выберите изображение для обработки",
                                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not image_path:
        print("Изображение не выбрано.")
        return

    # Выбор пути для сохранения маски
    output_mask_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                    title="Выберите место для сохранения маски",
                                                    filetypes=[("PNG files", "*.png")])
    if not output_mask_path:
        print("Место для сохранения маски не выбрано.")
        return

    # Загружаем модель и обрабатываем изображение
    model = load_model()
    process_image(image_path, output_mask_path, model)


# Интеграция с Photoshop через JSX
def apply_mask_in_photoshop(image_path, mask_path):
    jsx_script = f"""
    #target photoshop
    var maskFile = new File("{mask_path}");
    if (maskFile.exists) {{
        var maskDoc = open(maskFile);
        maskDoc.selection.selectAll();
        maskDoc.selection.copy();
        maskDoc.close(SaveOptions.DONOTSAVECHANGES);

        var doc = app.open(new File("{image_path}"));
        doc.paste();
        var maskLayer = doc.activeLayer;
        doc.selection.load(maskLayer, SelectionType.REPLACE);
        doc.activeLayer.grouped = true;
        maskLayer.remove();

        alert("Маска успешно применена!");
    }} else {{
        alert("Маска не найдена.");
    }}
    """
    jsx_file = "apply_mask.jsx"

    # Сохраняем JSX-скрипт во временный файл
    with open(jsx_file, "w") as f:
        f.write(jsx_script)

    # Запуск Photoshop с JSX-скриптом (адаптируйте для Windows)
    subprocess.run([r"D:\Adobe\Adobe Photoshop 2025\Photoshop.exe", "-r", os.path.abspath(jsx_file)])


# Основная функция
def main():
    run_interface()


if __name__ == "__main__":
    main()
