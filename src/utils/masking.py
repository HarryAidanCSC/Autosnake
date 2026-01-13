import cv2 as cv
from cv2.typing import MatLike


def apply_morpho_masking(mask: MatLike) -> MatLike:
    """Apply morphological operations to the mask.

    Args:
        mask (MatLike): Original mask.

    Returns:
        MatLike: Mask with morphological operations.
    """
    # Apply morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    mask = cv.GaussianBlur(mask, (5, 5), 0)

    return mask
