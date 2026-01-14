import cv2 as cv
import numpy as np
from typing import Tuple, Optional
from cv2.typing import MatLike, Point

from utils.masking import apply_morpho_masking


class ImageAnalyser:

    def __init__(self, greyscale: bool = False) -> None:
        self.GREYSCALE = greyscale

    def _read_image(self, file_path: str) -> MatLike:
        # Read image in greyscale
        image = cv.imread(file_path, int(not self.GREYSCALE))

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

    def _get_contours(
        self, frame: MatLike, lower_bound: list[int], upper_bound: list[int]
    ) -> Optional[Tuple[int, int, int, int, MatLike]]:
        """Detect an object by countouring in HSV space.

        Args:
            frame: BGR image frame
            lower_bound: Lower HSV bound as [H, S, V]
            upper_bound: Upper HSV bound as [H, S, V]

        Returns:
            (x, y, w, h) position and dimensions of the detected object, or None if not found.
        """
        # Create color mask
        lower_bound_np = np.array(lower_bound, dtype="uint8")
        upper_bound_np = np.array(upper_bound, dtype="uint8")

        # Initial mask
        mask = cv.inRange(frame, lower_bound_np, upper_bound_np)

        # Apply morphological operations to reduce noise
        mask = apply_morpho_masking(mask=mask)

        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter by area - minimum area threshold
        valid_contours = [c for c in contours if cv.contourArea(c) > 50]
        if not valid_contours:
            return None

        # Get largest contour
        largest_contour = max(valid_contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)

        return x, y, w, h, largest_contour

    def get_corners(
        self, frame: MatLike, lower_bound: list[int], upper_bound: list[int]
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        contours = self._get_contours(
            frame=frame, lower_bound=lower_bound, upper_bound=upper_bound
        )

        # Return None if no contours found
        if not contours:
            return None
        x, y, w, h, _ = contours

        top_left = (x, y)
        bottom_right = (x + w, y + h)

        return (top_left, bottom_right)

    def colour_filtering(
        self,
        frame: MatLike,
        lower_bound: list[int],
        upper_bound: list[int],
        is_snake_head=False,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect an object  by color filtering in HSV space.

        Args:
            frame: BGR image frame
            lower_bound: Lower HSV bound as [H, S, V]
            upper_bound: Upper HSV bound as [H, S, V]

        Returns:
            (x, y) position of the detected object's center, or None if not found
        """

        # First locate contours
        contours = self._get_contours(
            frame=frame, lower_bound=lower_bound, upper_bound=upper_bound
        )

        # Return None if no contours found
        if not contours:
            return None
        x, y, w, h, largest_contour = contours

        if w == 0 or h == 0:
            return None

        # Extract ROI for eye detection
        roi_hsv = frame[y : y + h, x : x + w]

        # Detect white/light colored eyes
        # Adjusted thresholds for better eye detection
        lower_bound_white = np.array([0, 0, 180], dtype="uint8")
        upper_bound_white = np.array([180, 60, 255], dtype="uint8")
        mask_eyes = cv.inRange(roi_hsv, lower_bound_white, upper_bound_white)

        # Clean up eye mask
        kernel_small = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
        mask_eyes = cv.morphologyEx(mask_eyes, cv.MORPH_OPEN, kernel_small)

        # Find eye contours
        contours_eyes, _ = cv.findContours(
            mask_eyes, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # Try to use eye position if detected
        if is_snake_head and contours_eyes:
            # Filter small contours that might be noise
            valid_eyes = [c for c in contours_eyes if cv.contourArea(c) > 5]

            if valid_eyes:
                # Calculate center of all eye contours
                all_eyes = np.concatenate(valid_eyes)
                M = cv.moments(all_eyes)

                if M["m00"] != 0:
                    cx_local = int(M["m10"] / M["m00"])
                    cy_local = int(M["m01"] / M["m00"])
                    return (x + cx_local, y + cy_local, w, h)

        #  Use the center of the largest contour
        M = cv.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy, w, h)

        # Final fallback: Use bounding box center
        return (x + w // 2, y + h // 2, w, h)

    def match_with_template(
        self, img: MatLike, comparison_frame: MatLike
    ) -> Tuple[float, Point]:
        """Match a frame with a template.

        Args:
            img (MatLike): Template to match
            comparison_frame (MatLike): Frame.

        Returns:
            Tuple[float, Point]: Value and location of frame
        """
        result = cv.matchTemplate(img, comparison_frame, cv.TM_CCOEFF_NORMED)

        # Closest match
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        return max_val, max_loc
