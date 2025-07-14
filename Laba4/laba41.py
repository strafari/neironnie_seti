import sys  # Импортируем модуль для взаимодействия с системой
import os  # Импортируем модуль для работы с файловой системой
import numpy as np  # Импортируем библиотеку для научных вычислений
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy  # Импортируем необходимые классы из PyQt5
from PyQt5.QtGui import QPixmap, QImage, QFont  # Импортируем необходимые классы для работы с графикой и шрифтами
from PyQt5.QtCore import Qt  # Импортируем класс для работы с основными объектами Qt

# Класс, реализующий сеть Хопфилда
class HopNET:
    def __init__(self, size, epsilon=1e-5):
        self.size = size  # Задаем размер сети
        self.weights = np.zeros((size, size))  # Инициализируем матрицу весов нулями
        self.epsilon = epsilon  # Устанавливаем пороговое значение для обучения

    def train(self, patterns):
        for pattern in patterns:  # Для каждого паттерна
            self.weights += np.outer(pattern, pattern)  # Обновляем матрицу весов
        np.fill_diagonal(self.weights, 0)  # Обнуляем диагональные элементы матрицы весов

    def energy(self, state):
        return -0.5 * np.dot(state.T, np.dot(self.weights, state))  # Вычисляем энергию текущего состояния

    def update(self, state):
        return np.sign(np.dot(self.weights, state))  # Обновляем состояние сети

    def run(self, state):
        previous_state = np.zeros_like(state)  # Инициализируем предыдущее состояние
        flag = True  # Флаг для выхода из цикла
        while flag:  # Пока состояние не стабилизируется
            new_state = self.update(state)  # Обновляем состояние
            if np.array_equal(new_state, previous_state):  # Если состояние не изменилось
                flag = False  # Выходим из цикла
            previous_state = state  # Обновляем предыдущее состояние
            state = new_state  # Устанавливаем новое состояние
        return state  # Возвращаем стабилизированное состояние

    def unlearn(self, pattern):
        self.weights -= np.outer(pattern, pattern)  # Обновляем матрицу весов, вычитая вклад паттерна
        np.fill_diagonal(self.weights, 0)  # Обнуляем диагональные элементы матрицы весов

# Функция для добавления шума в изображение
def add_noise(image, noise_level=0.1):
    noisy_image = image.copy()  # Копируем изображение
    num_noisy_pixels = int(noise_level * image.size)  # Определяем количество пикселей для добавления шума
    noisy_indices = np.random.choice(image.size, num_noisy_pixels, replace=False)  # Генерируем индексы шумных пикселей
    noisy_image[noisy_indices] *= -1  # Инвертируем значение пикселей
    return noisy_image  # Возвращаем изображение с шумом

# Класс, реализующий графический интерфейс приложения
class HopApp(QWidget):
    def __init__(self, network):
        super().__init__()  # Инициализируем базовый класс
        self.network = network  # Сохраняем экземпляр сети Хопфилда
        self.patterns = []  # Инициализируем список паттернов
        self.current_image = None  # Текущее изображение
        self.current_image_index = -1  # Индекс текущего изображения
        self.noise_level = 0.1  # Уровень шума
        self.initUI()  # Инициализируем пользовательский интерфейс
        self.load_images()  # Загружаем изображения

    def initUI(self):
        layout = QVBoxLayout()  # Создаем вертикальный макет
        self.label = QLabel(self)  # Создаем метку для отображения изображения
        layout.addWidget(self.label)  # Добавляем метку в макет

        hbox = QHBoxLayout()  # Создаем горизонтальный макет
        self.image_buttons = []  # Список кнопок для выбора изображений
        for i in range(3):  # Создаем три кнопки
            btn = QPushButton(f'Изображение {i + 1}', self)  # Создаем кнопку
            btn.clicked.connect(lambda _, x=i: self.select_image(x))  # Подключаем обработчик клика
            hbox.addWidget(btn)  # Добавляем кнопку в горизонтальный макет
            self.image_buttons.append(btn)  # Добавляем кнопку в список кнопок
        layout.addLayout(hbox)  # Добавляем горизонтальный макет в основной макет

        self.selected_image_label = QLabel('Выбрано изображение: Нет', self)  # Метка для отображения выбранного изображения
        self.selected_image_label.setFont(QFont('Arial', 24))  # Устанавливаем шрифт метки
        self.selected_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # Устанавливаем политику размера
        self.selected_image_label.setAlignment(Qt.AlignCenter)  # Устанавливаем выравнивание по центру
        layout.addWidget(self.selected_image_label)  # Добавляем метку в макет

        self.train_button = QPushButton('Обучить', self)  # Кнопка для обучения сети
        self.train_button.clicked.connect(self.train)  # Подключаем обработчик клика
        layout.addWidget(self.train_button)  # Добавляем кнопку в макет

        self.run_button = QPushButton('Запустить', self)  # Кнопка для запуска сети
        self.run_button.clicked.connect(self.run)  # Подключаем обработчик клика
        layout.addWidget(self.run_button)  # Добавляем кнопку в макет

        self.noise_button = QPushButton('Добавить шум', self)  # Кнопка для добавления шума
        self.noise_button.clicked.connect(self.add_noise)  # Подключаем обработчик клика
        layout.addWidget(self.noise_button)  # Добавляем кнопку в макет

        self.unlearn_button = QPushButton('Разобучить', self)  # Кнопка для разобучения сети
        self.unlearn_button.clicked.connect(self.unlearn)  # Подключаем обработчик клика
        layout.addWidget(self.unlearn_button)  # Добавляем кнопку в макет

        self.setLayout(layout)  # Устанавливаем макет для окна
        self.setWindowTitle('Сеть Хопфилда')  # Устанавливаем заголовок окна
        self.show()  # Показываем окно

    def load_images(self):
        folder = 'hob_image'  # Путь к папке с изображениями
        patterns = []  # Список паттернов
        for filename in os.listdir(folder):  # Проходим по всем файлам в папке
            if filename.endswith('.bmp'):  # Если файл имеет расширение .bmp
                filepath = os.path.join(folder, filename)  # Полный путь к файлу
                image = QImage(filepath)  # Загружаем изображение
                if image.width() == 20 and image.height() == 20:  # Проверяем размер изображения
                    pattern = self.image_to_pattern(image)  # Преобразуем изображение в паттерн
                    patterns.append(pattern)  # Добавляем паттерн в список
        self.patterns = patterns  # Сохраняем список паттернов
        print("Изображения загружены")  # Выводим сообщение о загрузке изображений

    def select_image(self, index):
        if index < len(self.patterns):  # Если индекс допустим
            self.current_image = self.patterns[index]  # Устанавливаем текущее изображение
            self.current_image_index = index  # Сохраняем индекс текущего изображения
            self.noise_level = 0.1  # Сбрасываем уровень шума
            self.selected_image_label.setText(f'Выбрано изображение: {index + 1}')  # Обновляем метку с информацией о выбранном изображении
            self.display_image(self.current_image)  # Отображаем текущее изображение

    def image_to_pattern(self, image):
        gray_image = image.convertToFormat(QImage.Format_Grayscale8)  # Конвертируем изображение в градации серого
        buffer = gray_image.bits()  # Получаем байтовый буфер изображения
        buffer.setsize(gray_image.byteCount())  # Устанавливаем размер буфера
        array = np.frombuffer(buffer, dtype=np.uint8).reshape((20, 20))  # Преобразуем буфер в массив numpy
        pattern = np.where(array > 127, 1, -1).flatten()  # Преобразуем массив в паттерн, где значения > 127 преобразуются в 1, а остальные в -1
        return pattern  # Возвращаем паттерн

    def train(self):
        self.network.train(self.patterns)  # Обучаем сеть по загруженным паттернам
        print("Обучение завершено")  # Выводим сообщение о завершении обучения

    def run(self):
        if self.current_image is not None:  # Если текущее изображение установлено
            result = self.network.run(self.current_image)  # Запускаем сеть для текущего изображения
            self.display_image(result)  # Отображаем результат

    def add_noise(self):
        if self.current_image is not None:  # Если текущее изображение установлено
            self.noise_level += 0.1  # Увеличиваем уровень шума
            noisy_image = add_noise(self.current_image, self.noise_level)  # Добавляем шум в текущее изображение
            self.display_image(noisy_image)  # Отображаем зашумленное изображение

    def unlearn(self):
        if self.current_image is not None:  # Если текущее изображение установлено
            self.network.unlearn(self.current_image)  # Разобучаем сеть по текущему изображению
            print("Разобучение завершено")  # Выводим сообщение о завершении разобучения


    def display_image(self, image):
        img = (image.reshape(20, 20) * 255).astype(np.uint8)  # Преобразуем изображение для отображения
        qimage = QImage(img, 20, 20, QImage.Format_Grayscale8)  # Создаем QImage из массива numpy
        pixmap = QPixmap.fromImage(qimage).scaled(100, 100, Qt.KeepAspectRatio)  # Создаем QPixmap и масштабируем его
        self.label.setPixmap(pixmap)  # Устанавливаем QPixmap в метку

if __name__ == '__main__':
    app = QApplication(sys.argv)  # Создаем экземпляр приложения

    epsilon = 1e-5  # Устанавливаем значение эпсилон
    network = HopNET(400, epsilon=epsilon)  # Создаем экземпляр сети Хопфилда для изображений 20x20
    ex = HopApp(network)  # Создаем экземпляр приложения
    sys.exit(app.exec_())  # Запускаем основной цикл приложения