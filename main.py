import tkinter as tk
from tkinter import filedialog

from PIL import Image
from PIL import ImageTk

from backend import MML


class Application:
    image_size = 224  # Размер картинок

    def __init__(self, path_model: str = '') -> None:
        self.window = tk.Tk()  # Создание основного окна
        self.path_img = ''
        self.frame_title = tk.Frame(
            self.window,  # Окно
            padx=10,  # Oтступ по горизонтали
            pady=10  # Oтступ по вертикали
        )
        self.frame_title.pack(expand=True)
        self.canvas = tk.Canvas(self.window, width=self.image_size, height=self.image_size)
        self.canvas.pack()
        self.frame_response = tk.Frame(
            self.window,  # Окно
            padx=10,  # Oтступ по горизонтали
            pady=10  # Oтступ по вертикали
        )
        self.frame_response.pack(expand=True)
        self.model = MML(path_model=path_model, shape=(self.image_size, self.image_size))

    def start(self) -> None:
        """Запуск приложения"""
        # Оформление окна:
        self.window.title('AAA')
        self.window.geometry('400x400')

        # Панель ввода:
        label_input = tk.Label(
            self.frame_title,
            text="Добро пожаловать!\nХотите определить возраст по фотографии?",
            # font=tk.font.Font(family="Helvetica", size=16, weight="bold")
        )
        label_input.grid(row=1)

        # Обработка изображения:
        self.load_and_display_image('start_img.png')
        button = tk.Button(
            self.window,
            text="Выбрать изображение",
            command=self.open_file_dialog
        )
        button.pack(pady=10)

        self.window.mainloop()

    def open_file_dialog(self) -> None:
        """Открывает меню выбора файла и запускает анализ"""
        self.path_img = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")]
        )
        if self.path_img:  # Если указан путь
            self.load_and_display_image(self.path_img)
            self.analyzer(self.path_img)
        else:
            self.load_and_display_image('start_img.png')

    def analyzer(self, path_img: str) -> None:
        """Анализ изображения и вывод"""
        # Предсказание:
        img = self.model.load_image(path_img)
        predicted = self.model.predict(img)  # Предсказание
        result = round(predicted.flatten()[0] * 116)  # Масштабирование

        # Панель вывода:
        message = f'Возраст: {result}'
        label_input = tk.Label(self.frame_response, text=message)
        label_input.grid(row=3)

    def load_and_display_image(self, file_path: str) -> None:
        """Згрузка и отображение картинки в приложении"""
        img = Image.open(file_path)  # Открыть изображение
        resize_img = img.resize((self.image_size, self.image_size))  # Ресайз
        img = ImageTk.PhotoImage(resize_img)
        self.canvas.create_image(
          0,
          self.canvas.winfo_reqheight(),
          anchor=tk.SW,
          image=img
        )  # Отображение картинки
        self.canvas.image = img


# Запуск приложения:
application = Application(path_model='19.5_Age_model.hdf5')
application.start()
