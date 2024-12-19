import numpy as np
import cv2


def average_side_len(polygon):
    side_lengths = [
        np.linalg.norm(polygon[i] - polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    ]

    # Compute the average side length
    return np.mean(side_lengths)


def blur_kernel_size(polygon):
    """
    Calculates the Gaussian blur kernel size based on the polygon size.

    Args:
        polygon (ndarray): Array of polygon points

    Returns:
        tuple: Odd-sized kernel.
    """

    average_dim = average_side_len(polygon)
    kernel_size = max(1, int(average_dim * 0.01) | 1)  # At least 1 and odd

    return kernel_size


def dilation_kernel(polygon):
    """
    Calculates the dilation kernel size based on the polygon size.

    Args:
        polygon (ndarray): Array of polygon points

    Returns:
        ndarray: kernel for dilation.
    """

    average_dim = average_side_len(polygon)

    # Define kernel size as a fraction of the average side length
    # Fraction value was choosen empirically (about 1%)
    kernel_size = max(1, int(average_dim * 0.01))
    return np.ones((kernel_size, kernel_size), np.uint8)


def create_expanded_mask(image, polygon):
    """
    Generates expanded and blurred mask for a given barcode image

    Args:
        polygon (ndarray): Array of polygon points
        image (ndarray): The source image (BGR format).

    Returns:
        ndarray: mask for barcode replacement.
    """
    # Create a mask for the barcode area
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(int), 255)

    # Expand mask a little
    expanded_mask = np.zeros_like(mask)
    cv2.dilate(mask, dilation_kernel(polygon), dst=expanded_mask)

    # Apply blurring for smoother replacement
    kernel_size = blur_kernel_size(polygon)
    return cv2.GaussianBlur(expanded_mask, (kernel_size, kernel_size), 0)


def get_background_and_foreground(image, polygon):
    """
    Extracts foreground and background masks from a region.

    Args:
        image (ndarray): The source image (BGR format).
        polygon (ndarray): The polygon defining the barcode area.

    Returns:
        tuple: (background_mask, foreground_mask).
    """
    init_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(init_mask, polygon.astype(int), 255)
    mask = np.zeros_like(init_mask)
    cv2.dilate(init_mask, np.ones((3, 3), np.uint8), dst=mask)
    border_mask = mask & cv2.bitwise_not(init_mask)

    # Extract the ROI based on the mask
    roi = cv2.bitwise_and(image, image, mask=mask)

    # Convert to grayscale within the ROI for thresholding
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # We have two regions: the foreground and background
    foreground_mask = (binary == 0) & (mask == 255)
    background_mask = (binary == 255) & (mask == 255)

    # The background mask is the one, that has the most intersection with border_mask
    if cv2.countNonZero(255 * (foreground_mask & border_mask)) > cv2.countNonZero(
        255 * (background_mask & border_mask)
    ):
        background_mask, foreground_mask = foreground_mask, background_mask

    return background_mask, foreground_mask


def extract_colors(image, polygon):
    """
    Extracts the dominant foreground and background colors from a region.

    Args:
        image (ndarray): The source image (BGR format).
        polygon (ndarray): The polygon defining the barcode area.

    Returns:
        tuple: (background_color, foreground_color) as BGR tuples.
    """
    background_mask, foreground_mask = get_background_and_foreground(image, polygon)

    # Now extract the foreground and background colors
    foreground_color = np.mean(
        image[foreground_mask], axis=0
    )  # Average color of the foreground
    background_color = np.mean(
        image[background_mask], axis=0
    )  # Average color of the background

    return background_color, foreground_color


def recolor_barcode(image, polygon, background_color, foreground_color):
    """
    Recolors barcode on image to a given color scheme.
    It's better for image to be without noise, shadows, etc.
    (i.e. image should be synthetic, model etc.)
    It's needed so found masks would be more accurate

    Args:
        image (ndarray): The source image (BGR format).
        polygon (ndarray): The polygon defining the barcode area.
        background_color (BGR): New color for background.
        foreground_color (BGR): New color for bars.

    Returns:
        ndarray: New image of barcode with new colors
    """

    background_mask, foreground_mask = get_background_and_foreground(image, polygon)
    recolored = image.copy()

    background_color = np.array(background_color, dtype=np.uint8)
    foreground_color = np.array(foreground_color, dtype=np.uint8)
    recolored[background_mask] = background_color
    recolored[foreground_mask] = foreground_color

    return recolored


def correct_colors(base_image, base_polygon, new_image, new_polygon):
    """
    Recolors barcode on new_image to a color scheme of base_image's barcode.
    It's better for image to be without noise, shadows, etc.
    (i.e. image should be synthetic, model etc.)
    It's needed so found masks would be more accurate

    Args:
        base_image (ndarray): The source image (BGR format).
        base_polygon (ndarray): Polygon defining the area of original barcode.
        new_image (ndarray): Image with new barcode (BGR format).
        new_polygon (ndarray): Polygon defining the area of new barcode.

    Returns:
        ndarray: Image of new_image's barcode with base_image's colors
    """
    background_color, foreground_color = extract_colors(base_image, base_polygon)
    return recolor_barcode(new_image, new_polygon, background_color, foreground_color)
