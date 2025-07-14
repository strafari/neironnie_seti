import tkinter as tk  # Импортируем библиотеку tkinter для создания графического интерфейса
from tkinter import Canvas, messagebox  # Импортируем необходимые классы из tkinter
from PIL import Image, ImageDraw  # Импортируем классы Image и ImageDraw из библиотеки Pillow (PIL)
import numpy as np  # Импортируем библиотеку numpy для работы с массивами
import os  # Импортируем модуль os для работы с операционной системой

class NumberProcessor():  # Объявляем класс NumberProcessor для обработки цифр
    def __init__(self, folder):  # Определяем метод инициализации класса
        self.folder = folder  # Папка с изображениями
        self.learning_rate = 0.02  # Скорость обучения
        self.input_size = 10000  # Размер входного слоя (100x100 изображение -> 10000 пикселей)
        self.hidden_size = 1024  # Размер скрытого слоя
        self.num_classes = 10  # Количество классов (цифр 0-9)

        # Инициализация весов и смещений случайными значениями
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.num_classes) * 0.01
        self.b2 = np.zeros((1, self.num_classes))

        # Загрузка изображений и меток
        self.images, self.labels = self.load_images(folder=self.folder)
        # Обучение модели
        self.train(self.images, self.labels)

    def load_images(self, folder=None):  # Метод загрузки изображений и меток
        if folder is None:
            folder = self.folder
        images = []  # Список для хранения изображений
        labels = []  # Список для хранения меток
        for filename in os.listdir(folder):  # Проходим по файлам в папке
            img_path = os.path.join(folder, filename)  # Полный путь к изображению
            try:
                with Image.open(img_path) as img:  # Открываем изображение
                    img = img.convert("L")  # Преобразуем в оттенки серого
                    img = img.resize((100, 100))  # Изменяем размер изображения
                    images.append(np.asarray(img).flatten() / 255.0)  # Добавляем изображение в список
                    label = int(filename[0])  # Метка - цифра из названия файла
                    labels.append(label)  # Добавляем метку в список
            except Exception as e:  # Обрабатываем возможные ошибки
                print(f"Ошибка загрузки {img_path}: {e}")
        return np.array(images), np.array(labels)  # Возвращаем массивы изображений и меток

    def forward(self, X):  # Метод прямого прохода по сети
        Z1 = np.dot(X, self.W1) + self.b1  # Линейная комбинация и активация скрытого слоя
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(A1, self.W2) + self.b2  # Линейная комбинация выходного слоя
        A2 = self.softmax(Z2)  # Применяем функцию softmax
        return Z1, A1, Z2, A2  # Возвращаем активации и выходы слоев

    def softmax(self, x):  # Функция softmax
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def backward(self, X, Y, Z1, A1, Z2, A2):  # Метод обратного распространения ошибки
        m = Y.shape[0]

        dZ2 = A2 - Y  # Градиент функции потерь по выходу
        dW2 = np.dot(A1.T, dZ2) / m  # Градиент по весам выходного слоя
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # Градиент по смещениям выходного слоя

        dA1 = np.dot(dZ2, self.W2.T)  # Градиент по активациям скрытого слоя
        dZ1 = dA1 * (Z1 > 0)  # Обратное распространение через активацию ReLU
        dW1 = np.dot(X.T, dZ1) / m  # Градиент по весам скрытого слоя
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # Градиент по смещениям скрытого слоя

        return dW1, db1, dW2, db2  # Возвращаем градиенты

    def update_parameters(self, dW1, db1, dW2, db2):  # Метод обновления параметров сети
        self.W1 -= self.learning_rate * dW1  # Обновление весов скрытого слоя
        self.b1 -= self.learning_rate * db1  # Обновление смещений скрытого слоя
        self.W2 -= self.learning_rate * dW2  # Обновление весов выходного слоя
        self.b2 -= self.learning_rate * db2  # Обновление смещений выходного слоя

    def guess(self, image):  # Метод для предсказания класса по изображению
        _, _, _, A2 = self.forward(image)  # Прямой проход по сети
        return A2, np.argmax(A2, axis=1)[0]  # Возвращаем выход сети и предсказанный класс

    def train(self, images, labels):  # Метод обучения сети
        Y = np.eye(self.num_classes)[labels]  # Преобразуем метки в one-hot векторы
        epoch = 0
        max_epochs = 2200
        flag=False
        while epoch < max_epochs and not flag:  # Цикл обучения
            Z1, A1, Z2, A2 = self.forward(images)  # Прямой проход
            loss = self.cross_entropy_loss(Y, A2)  # Вычисляем функцию потерь
            max_loss = max(loss)  # Средняя потеря по всем образцам
            dW1, db1, dW2, db2 = self.backward(images, Y, Z1, A1, Z2, A2)  # Обратный проход
            self.update_parameters(dW1, db1, dW2, db2)  # Обновление параметров
            print(f"Эпоха {epoch}, максимальное потери: {max_loss}")  # Выводим информацию о средних потерях
            if max_loss < 1.5:  # Прекращаем обучение, если достигнута необходимая точность
                flag=True
            epoch += 1
        print(f"Обучение завершено после {epoch} эпох")  # Сообщаем о завершении обучения

    def cross_entropy_loss(self, y_true, y_pred):  # Функция потерь
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # Защита от деления на ноль
        n_samples = y_true.shape[0]
        log_p = -np.log(y_pred[np.arange(n_samples), y_true.argmax(axis=1)])  # Логарифм правильных предсказаний
        return log_p  # Возвращаем массив ошибок правильных предсказаний для каждого образца


    def save_weights(self):  # Метод сохранения весов сети в файл
        np.savez("weights.npz", W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

class TestCanvas(tk.Tk):  # Объявляем класс TestCanvas для создания графического интерфейса
    def __init__(self, processor):  # Определяем метод инициализации класса
        super().__init__()  # Вызываем метод инициализации родительского класса
        self.processor = processor  # Экземпляр класса NumberProcessor
        self.canvas = Canvas(self, width=280, height=280, bg="white")  # Создаем холст tkinter
        self.canvas.pack()  # Размещаем холст на главном окне
        self.bind("<B1-Motion>", self.draw)  # Привязываем событие рисования на холсте к методу draw
        self.image = Image.new("L", (280, 280), "white")  # Создаем изображение PIL
        self.picture_draw = ImageDraw.Draw(self.image)  # Создаем объект для рисования на изображении
        test_button = tk.Button(self, text="Распознание числа", command=self.test_image)  # Создаем кнопку "Распознание числа"
        test_button.pack(side=tk.BOTTOM)  # Размещаем кнопку внизу окна
        clear_button = tk.Button(self, text="Очистить", command=self.clear_canvas)  # Создаем кнопку "Очистить"
        clear_button.pack(side=tk.BOTTOM)  # Размещаем кнопку внизу окна
        save_weights = tk.Button(self, text="Сохранить весовые коэффициенты", command=self.save_w)  # Создаем кнопку "Сохранить весовые коэффициенты"
        save_weights.pack(side=tk.BOTTOM)  # Размещаем кнопку внизу окна
        research = tk.Button(self, text="Тестирование", command=self.research)  # Создаем кнопку "Тестирование"
        research.pack(side=tk.BOTTOM)  # Размещаем кнопку внизу окна

    def draw(self, event):  # Метод для обработки события рисования на холсте
        x, y = event.x, event.y  # Получаем координаты точки рисования
        self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="black")  # Рисуем круг на холсте
        self.picture_draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill="black")  # Рисуем круг на изображении

    def test_image(self):  # Метод для тестирования изображения
        resized_image = self.image.resize((100, 100))  # Изменяем размер изображения
        img_array = np.array(resized_image) / 255.0  # Преобразуем изображение в массив и нормализуем
        img_array = img_array.flatten()  # Выравниваем массив
        result, flag = self.processor.guess(img_array)  # Предсказываем цифру на изображении
        messagebox.showinfo("Результат", f"Это похоже на цифру {flag}!")  # Выводим результат в сообщении
        self.clear_canvas()  # Очищаем холст

    def save_w(self):  # Метод для сохранения весов сети
        self.processor.save_weights()  # Вызываем метод сохранения весов
        messagebox.showinfo("Результат", "Весовые коэффициенты сохранены!")  # Выводим сообщение об успешном сохранении

    def clear_canvas(self):  # Метод для очистки холста
        self.canvas.delete("all")  # Удаляем все объекты на холсте
        self.image = Image.new("L", (280, 280), "white")  # Создаем новое белое изображение
        self.picture_draw = ImageDraw.Draw(self.image)  # Создаем объект для рисования на новом изображении

    def research(self):  # Метод для тестирования нейронной сети на тестовом наборе
        test_folder = "test"  # Папка с тестовыми изображениями
        re_images, re_labels = self.processor.load_images(test_folder)  # Загружаем тестовые изображения и метки
        wrong_count = 0  # Счетчик неправильно распознанных изображений
        for img, label in zip(re_images, re_labels):  # Проходим по тестовым изображениям
            _, flag = self.processor.guess(img)  # Предсказываем класс
            if flag is None or flag != label:  # Если предсказание неверное
                wrong_count += 1  # Увеличиваем счетчик ошибок
        messagebox.showinfo(  # Выводим результат тестирования в сообщении
            "Результат",
            f"Количество ошибок: {wrong_count}, процент не распознанных образов: {(wrong_count/len(re_labels))*100:.2f}%"
        )

if __name__ == "__main__":  # Точка входа в программу
    folder_path = "learn"  # Путь к папке с обучающими изображениями
    processor = NumberProcessor(folder_path)  # Создаем экземпляр класса NumberProcessor
    app = TestCanvas(processor)  # Создаем экземпляр класса TestCanvas
    app.title("Распознавание чисел")  # Устанавливаем заголовок окна
    app.mainloop()  # Запускаем главный цикл программы 