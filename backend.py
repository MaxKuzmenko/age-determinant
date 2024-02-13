import numpy as np
import tensorflow as tf


class MML:
    image_size = 224  # Размер картинок для анализа

    def __init__(self, path_model: str, shape: tuple[int, int] = (1, 1)) -> None:
        self.img = None
        self.new_shape = shape
        self.model = tf.keras.saving.load_model(path_model)  # Загрузка модели

    def load_image(self, path_to_img: str) -> np.array:
        """Считывание изображения"""

        # Считываени файла:
        img = tf.io.read_file(path_to_img)  # Открыть изображение
        img = tf.image.decode_image(img, channels=3)  # Декодинг
        img = tf.image.resize(img, self.new_shape)  # Ресайз
        img = img[tf.newaxis, :]  # Добавление batch dimension
        return img

    def predict(self, img: np.array) -> np.array:
        """Выполняет анализ картинки моделью"""
        img = self.preprocessing(img, data_format='channels_last')
        predicted = self.model.predict(img)
        return predicted

    def preprocessing(self, img, data_format=None, version=2) -> np.array:
        """"""
        img = np.copy(img)

        if version == 1:
            if data_format == 'channels_first':
                x_temp = img[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 93.5940
                x_temp[:, 1, :, :] -= 104.7624
                x_temp[:, 2, :, :] -= 129.1863
            else:
                x_temp = img[..., ::-1]
                x_temp[..., 0] -= 93.5940
                x_temp[..., 1] -= 104.7624
                x_temp[..., 2] -= 129.1863

        elif version == 2:
            if data_format == 'channels_first':
                x_temp = img[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 91.4953
                x_temp[:, 1, :, :] -= 103.8827
                x_temp[:, 2, :, :] -= 131.0912
            else:
                x_temp = img[..., ::-1]
                x_temp[..., 0] -= 91.4953
                x_temp[..., 1] -= 103.8827
                x_temp[..., 2] -= 131.0912
        else:
            raise NotImplementedError
        return x_temp

    def decode(self, img):
        """"""
        pass
