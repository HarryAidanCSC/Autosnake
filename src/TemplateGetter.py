import cv2 as cv
import numpy as np
from typing import Tuple
from cv2.typing import MatLike


class TemplateGetter:

    def __init__(self, greyscale: bool = True) -> None:
        self.greyscale = greyscale

    def _read_image(self, file_path: str) -> MatLike:
        # Read image in greyscale
        image = cv.imread(file_path, int(not self.greyscale))

        if image is None:
            raise FileNotFoundError(
                f"Error: Could not find {file_path}. Please take a screenshot and crop it first!"
            )

        return image

    def get_image_and_rotations(self, file_path: str) -> list[Tuple[MatLike, int, int]]:
        image = self._read_image(file_path=file_path)

        images = []
        for _ in range(4):
            h, w = image.shape[:2]
            images.append((image, w, h))
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

        return images

    def get_image_and_scaled(
        self, file_path: str, scales: list[float] = [0.8, 1.0, 1.2]
    ) -> list[Tuple[MatLike, int, int]]:
        image = self._read_image(file_path=file_path)
        h, w = image.shape[:2]

        images = []
        for scale in scales:
            new_width = int(w * scale)
            new_height = int(h * scale)
            resized_image = cv.resize(image, (new_width, new_height))
            images.append((resized_image, w, h))

        return images
