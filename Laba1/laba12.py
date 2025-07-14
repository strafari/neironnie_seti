import numpy as np
import os
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Canvas, messagebox

# Инициализация параметров
LEARNING_RATE = 0.1
IMAGE_SIZE = (100, 100)

# Инициализация весов
weights = np.random.uniform(-0.3, 0.3, (10, *IMAGE_SIZE))

def load_images(folder):
    """Загрузка изображений и меток из папки."""
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.convert("L").resize(IMAGE_SIZE)
                img = img.point(lambda x: 1 if x < 128 else 0)
                images.append(np.asarray(img))
                labels.append(int(filename[0]))
        except Exception as e:
            print(f"Ошибка загрузки {img_path}: {e}")
    return np.array(images), np.array(labels)

def threshold_activation(x):
    """Функция активации"""
    return 1 if x >= 0 else 0

def predict(input_data):
    """Предсказание метки для входных данных"""
    weighted_sum = [np.dot(weights[i].flatten(), input_data.flatten()) for i in range(10)]
    outputs = np.array([threshold_activation(x) for x in weighted_sum])
    
    if np.sum(outputs) != 1:
        return None  # Ошибка: активировалось более одного или ни одного нейрона
    return np.argmax(outputs)

def train(images, labels):
    """Обучение нейронной сети"""
    global weights
    epoch = 0

    while True:
        errors = 0
        # Перемешивание данных
        indices = np.random.permutation(len(images))
        images, labels = images[indices], labels[indices]

        for img, label in zip(images, labels):
            weighted_sum = [np.dot(weights[i].flatten(), img.flatten()) for i in range(10)]
            outputs = np.array([threshold_activation(x) for x in weighted_sum])

            # Обновление весов
            for n in range(10):
                error = (1 if n == label else 0) - outputs[n]
                if error != 0:
                    weights[n] += LEARNING_RATE * error * img

            # Проверка предсказания
            pred = predict(img)
            if pred != label:
                errors += 1

        epoch += 1
        print(f"Эпоха {epoch}: ошибок {errors}")

        if errors == 0:  # Завершаем, если ошибок нет
            print(f"Обучение завершено за {epoch} эпох")
            return


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
        graf =  tk.Button(self, text="Показать графики", command=self.graf)
        graf.pack(side=tk.BOTTOM)
        
        
    def draw(self, event):
        x, y = event.x, event.y  
        self.canvas.create_oval(x-7, y-7, x+7, y+7, fill='black')  
        self.draw.ellipse([x-7, y-7, x+7, y+7], fill='black') 

    def test_image(self):
        # Инвертируем изображение, изменяем размер и преобразуем его в двоичный формат
        inverted_image = ImageOps.invert(self.image)
        inverted_image = inverted_image.resize((100, 100))
        img_array = np.array(inverted_image)
        img_array = (img_array > 128).astype(int)  # Пороговая обработка и конвертация в 0 и 1
        
        # Предсказываем результат на основе обработанного изображения
        result = predict(img_array)
        if result == None:
            messagebox.showinfo("Результат", f"Это не похоже не на одну из цифр!")
        else:
            messagebox.showinfo("Результат", f"Это похоже на цифру {result}!")
        
        self.clear_canvas()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

   
    def graf(self):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))  # Подготовка сетки подграфиков
        fig.suptitle('Веса нейронов нейронной сети', fontsize=16)

        for i, ax in enumerate(axes.flatten()):
            if i < len(weights):
                cax = ax.imshow(weights[i].reshape((100, 100)), cmap='GnBu')  # Визуализация весов нейрона i
                ax.set_title(f"Нейрон {i}")
                ax.axis('off')  # Убрать оси на графиках
        fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='horizontal')
        plt.show()

    # Добавление цветовой шкалы и отображение графиков
    def research(self):
        folder_path = "test"
        re_images, re_labels = load_images(folder_path)
        wrong_count = 0
        for img, label in zip(re_images, re_labels):
            pred = predict(img)
            if pred == None:
                wrong_count += 1
            elif pred != label:
                wrong_count += 1
        messagebox.showinfo("Результат", f"Кол-во ошибок:{wrong_count},процент ошибок:{(wrong_count/100)*100}")

if __name__ == "__main__":
    folder_path = "learn"
    images, labels = load_images(folder_path)
    train(images, labels)
    app = TestCanvas()
    app.title("Тест нейронной сети: Распознавание цифр")
    app.mainloop()

