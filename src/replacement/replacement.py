import numpy as np
import cv2
import src.replacement.details as details
from enum import Enum


class Mode(Enum):
    # In this mode seamless_replace won't try to adjust color schemes of barcodes
    NO_AUTOCOLOR = 0
    # Automatically adjust colors of barcodes
    AUTOCOLOR = 1


def seamless_replace(
    base_image, base_polygon, new_image, new_polygon, mode=Mode.AUTOCOLOR
):
    """
    Replaces barcode on base_image with a barcode on new_image.
    For now, base_polygon and new_polygon should have exactly 4 vertices!
    It's better for new_image to be without noise, shadows, etc.
    (i.e. new_image should be synthetic, model etc.)


    Args:
        base_image (ndarray): The source image (BGR format).
        base_polygon (ndarray): Quadrilateral defining the area of original barcode.
        new_image (ndarray): Image with new barcode (BGR format).
        new_polygon (ndarray): Quadrilateral defining the area of new barcode.
        mode (Mode): The mode of seamless_replace (NO_AUTOCOLOR/AUTOCOLOR for now)

    Returns:
        ndarray: Image of barcode from new_image pasted onto base_image
    """

    if mode == Mode.AUTOCOLOR:
        new_image = details.correct_colors(base_image, base_polygon, new_image, new_polygon)

    # Transform new barcode into original's perspective
    M = cv2.getPerspectiveTransform(new_polygon, base_polygon)
    new_image_warped = cv2.warpPerspective(
        new_image, M, (base_image.shape[1], base_image.shape[0])
    )

    # Seamless cloning
    mask = details.create_expanded_mask(base_image, base_polygon)
    center = tuple(np.mean(base_polygon, axis=0).astype(int))

    seamless_result = cv2.seamlessClone(
        new_image_warped, base_image, mask, center, cv2.NORMAL_CLONE
    )

    return seamless_result
