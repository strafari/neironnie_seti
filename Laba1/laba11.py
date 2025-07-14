import numpy as np
import os
import tkinter as tk
from tkinter import Canvas, messagebox
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Функция для получения границ объекта на изображении
def get_object_bounds(image):

    image_array = np.array(image)  
    
    # Получение ненулевых столбцов и строк
    non_empty_columns = np.where(image_array.min(axis=0) < 255)[0]
    non_empty_rows = np.where(image_array.min(axis=1) < 255)[0]
    
    # Если есть ненулевые строки и столбцы, то можно определить границы объекта
    if non_empty_columns.any() and non_empty_rows.any():
        upper, lower = non_empty_rows[0], non_empty_rows[-1]  # Верхняя и нижняя границы
        left, right = non_empty_columns[0], non_empty_columns[-1]  # Левая и правая границы
        return left, upper, right, lower  # кортеж с границами объекта
    else:
        return None  

# Функция для центрирования объекта на изображении
def center_object(image):

    bounds = get_object_bounds(image)  
    if bounds:
        left, upper, right, lower = bounds  
        object_width = right - left  # Ширина объекта
        object_height = lower - upper  # Высота объекта
        horizontal_padding = (image.width - object_width) // 2  # Горизонтальные отступы для центрирования
        vertical_padding = (image.height - object_height) // 2  # Вертикальные отступы для центрирования
        
        cropped_image = image.crop(bounds)  # Обрезает изображение по найденным границам
        centered_image = Image.new("L", (image.width, image.height), "white")  # Создание пустого изображения
        centered_image.paste(cropped_image, (horizontal_padding, vertical_padding))  # Вставляет обрезанное изображение в центр
        return centered_image 
    return image  # Если объект не найден, возвращаем исходное изображение

# Функция для загрузки изображений и меток из папки
def load_images(folder):

    images = []  # Список для хранения изображений
    labels = []  # Список для хранения меток
    for filename in os.listdir(folder):  # Проходится по всем файлам в папке
        img_path = os.path.join(folder, filename)  # Путь к файлу изображения
        try:
            with Image.open(img_path) as img:
                img = img.convert("L")  # Преобразование изображение в оттенки серого
                img = center_object(img)  # Центрирование изображение
                img = img.resize((100, 100))  # Изменение размера изображения
                images.append(np.asarray(img).flatten() / 255.0)  # Преобразование изображение в массив и нормализуем значения
                label = int(filename[0])  # Извлечение метки из имени файла
                labels.append(label)  # Добавление метки в список
        except Exception as e:
            print(f"Ошибка загрузки {img_path}: {e}")  
    return np.array(images), np.array(labels)  

#Функция активации
def threshold(value, threshold=0):
    """Возвращает 1, если значение больше порога, иначе 0."""
    return 1 if value > threshold else 0

def guess(image, weights, x=0):
    """
    Предсказывает класс изображения.
    """
    activations = [np.dot(w, image) for w in weights]
    output = [threshold(a, x) for a in activations]
    predicted_classes = [i for i, o in enumerate(output) if o == 1]

    # Если предсказание однозначное (только один активный класс), вернуть его
    if len(predicted_classes) == 1:
        return output, predicted_classes[0]
    else:
        return output, None  # Неоднозначное предсказание


# Функция для вычисления количества ошибок
def calculate_errors(images, labels, weights):
    """
    Эта функция вычисляет количество ошибок для заданных данных, 
    используя предсказания модели.
    """
    wrong_count = 0  # Счетчик ошибок
    for img, label in zip(images, labels):  # Проходится по всем изображениям и меткам
        flag = guess(img, weights, 0)  # Делает предсказание
        if flag is None or flag != label:  # Если предсказание не совпадает с меткой или неоднозначно
            wrong_count += 1  # Увеличивает счетчик ошибок
    return wrong_count  # Возвращает количество

# Функция для обучения модели
def train(images, labels, weights, learning_rate, tolerance = 0):
    """
    Эта функция обучает модель с использованием алгоритма перцептрона.
    """
    epoch = 0  # Счетчик эпох
    while True:
        indices = np.random.permutation(len(images))  # Перемешивает индексы изображений
        images_shuffled = images[indices]  # Перемешивает изображения
        labels_shuffled = labels[indices]  # Перемешивает метки
        errors = 0  # Счетчик ошибок
        
        for img, label in zip(images_shuffled, labels_shuffled):  # Проходит по изображениям и меткам
            predictions, predicted_label = guess(img, weights, x=0)  # Делает предсказание
            if predicted_label != label:  # Если предсказание неверное
                errors += 1  # Увеличивает счетчик ошибок
                weights[label] += learning_rate * img  # Усиливает вес правильного класса
                for i in range(len(weights)):  # Проходит по всем классам
                    if predictions[i] == 1 and i != label:  # Если класс неверно предсказан
                        weights[i] -= learning_rate * img  # Ослабляет вес неверного класса
        epoch += 1  
        
        if errors <= tolerance:  # Если ошибок меньше или равно допустимому порогу
            print(f"Обучение завершено на эпохе {epoch}. Количество ошибок: {errors}")
            return  # Завершает обучение
        print(f"Эпоха {epoch}; Ошибок: {errors}")  
        
# Класс для графического интерфейса тестирования нейронной сети
class TestCanvas(tk.Tk):
    def __init__(self):
        super().__init__()
        self.canvas = Canvas(self, width=300, height=300, bg="white")
        self.canvas.pack()
        self.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        test_button = tk.Button(self, text="Проверить", command=self.test_image)
        test_button.pack(side=tk.BOTTOM)
        clear_button = tk.Button(self, text="Очистить", command=self.clear_canvas)
        clear_button.pack(side=tk.BOTTOM)
        research = tk.Button(self, text="ОШИБКИ", command=self.research)
        research.pack(side=tk.BOTTOM)
        graf =  tk.Button(self, text="Показать графики", command=self.show_weights_graph)
        graf.pack(side=tk.BOTTOM)
        
        
    def draw(self, event):
        x, y = event.x, event.y  
        self.canvas.create_oval(x-7, y-7, x+7, y+7, fill='black')  
        self.draw.ellipse([x-7, y-7, x+7, y+7], fill='black') 

    def test_image(self):

        centered_image = center_object(self.image)  
        resized_image = centered_image.resize((28, 28))  
        img_array = np.array(resized_image).flatten() / 255.0  

        result, flag = guess(img_array, weights, x=0)  # Делаем предсказание
        if flag is None:
            messagebox.showinfo("Результат", "Не похоже на одну из цифр!")  # Если предсказание не удалось
        else:
            messagebox.showinfo("Результат", f"Похоже на цифру {flag}!")  # Результат предсказания
        self.clear_canvas()  
    
    # Функция для тестирования на тестовой выборке
    def research(self):
        folder_path = "test"
        re_images, re_labels = load_images(folder_path)
        wrong_count = 0
        for img, label in zip(re_images, re_labels):
            pred, flag = guess(img, weights, 0)
            if flag == None:
                wrong_count += 1
            elif flag != label:
                wrong_count += 1
        messagebox.showinfo("Результат", f"Кол-во ошибок:{wrong_count},процент ошибок:{(wrong_count/100)*100}")


    def clear_canvas(self):
        
        self.canvas.delete("all")  
        self.image = Image.new("L", (280, 280), "white")  
        self.draw = ImageDraw.Draw(self.image)  

    def show_weights_graph(self):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8)) 
        for i, ax in enumerate(axes.flat): 
            ax.imshow(weights[i].reshape((100, 100)), cmap='GnBu') 
            ax.set_title(f"Веса класса {i}")  
        plt.tight_layout()  
        plt.show()  


# Главная программа
if __name__ == "__main__":
    folder_path = "learn"  # Путь к папке с обучающими данными
    weights = np.random.uniform(-0.3, 0.3, (10, 10000))  # Инициализация весов для 10 классов
    learning_rate = 0.01  # Установка скорости обучения
    images, labels = load_images(folder_path)  # Загружает данные и обучаем модель
    train(images, labels, weights, learning_rate)  # Запуск обучения
    print("Обучение завершено.")
    
    # Запуск графического интерфейса
    app = TestCanvas()  
    app.title("Тест нейронной сети")
    app.mainloop()  # Запуск интерфейса
