
import json
import numpy as np
import cv2

from sklearn.cluster import KMeans

def ensure_quadrilateral(polygon):
    hull = cv2.convexHull(polygon)

def blur_kernel_size(bounding_box):
    side_lengths = [
        np.linalg.norm(bounding_box[i] - bounding_box[(i + 1) % len(bounding_box)])
        for i in range(len(bounding_box))
    ]
    
    # Compute the average side length
    average_dim = int(np.mean(side_lengths))
    kernel_size = max(3, (average_dim // 10) | 1)  # At least 3 and odd

    return kernel_size

def dilation_kernel(polygon):
    """
    Dynamically calculates the dilation kernel size based on the polygon size.
    
    Args:
        polygon (ndarray): Array of polygon points
    
    Returns:
        tuple: Odd-sized kernel (width, height) for dilation.
    """
    # Calculate the lengths of the polygon's sides
    side_lengths = [
        np.linalg.norm(polygon[i] - polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    ]
    
    # Compute the average side length
    avg_side_length = np.mean(side_lengths)

    # Define kernel size as a fraction of the average side length
    # Fraction value was choosen empirically (about 8.5%)
    kernel_size = int(avg_side_length * 0.085)
    
    return np.ones((kernel_size, kernel_size), np.uint8)


def create_expanded_mask(polygon, img):
    # Create a mask for the barcode area
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(int), (255, 255, 255))
    
    # Expand mask a little
    expanded_mask = np.zeros_like(mask)
    cv2.dilate(mask, dilation_kernel(polygon), dst=expanded_mask)
    
    # Apply blurring for smoother replacement 
    kernel_size = blur_kernel_size(polygon)
    return cv2.GaussianBlur(expanded_mask, (kernel_size, kernel_size), 0)


def extract_colors(image, polygon):
    """
    Extracts the dominant foreground and background colors from a region.
    
    Args:
        image (ndarray): The source image (BGR format).
        polygon (ndarray): The quadrilateral defining the barcode area.
    
    Returns:
        tuple: (background_color, foreground_color) as BGR tuples.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(int), 255)

    # Extract the ROI based on the mask
    roi = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to grayscale within the ROI for thresholding
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    
    # Find connected components
    _, labels, _, _ = cv2.connectedComponentsWithStats(edges, 8, cv2.CV_32S)

    # We have two regions: the foreground and background. The background has index 0
    background_label = 0
    foreground_region = np.zeros_like(roi)
    foreground_region[labels == background_label] = 255
    fr = roi[foreground_region > 0]
    print(fr)
    # Now extract the foreground and background colors
    foreground_avg_color = np.mean(roi[foreground_region > 0], axis=0)  # Average color of the foreground
    background_avg_color = np.mean(roi[foreground_region == 0], axis=0)  # Average color of the background

    return foreground_avg_color, background_avg_color



def seamless_replace(base_img, src_polygon, new_barcode, dst_polygon):
    M = cv2.getPerspectiveTransform(dst_polygon, src_polygon)
    warped_barcode  = cv2.warpPerspective(new_barcode, M, (base_img.shape[1], base_img.shape[0]))
    
    mask = create_expanded_mask(src_polygon, base_img)

    # Seamless cloning
    center = tuple(np.mean(src_polygon, axis=0).astype(int))
    seamless_result = cv2.seamlessClone(
        warped_barcode, base_img, mask, center, cv2.NORMAL_CLONE
    )

    cv2.imshow("Seamless Barcode Replacement", seamless_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_color(color, window_name="Color"):
    """
    Displays a single color as an image using OpenCV.

    Args:
        color (tuple): The BGR color to display.
        window_name (str): Name of the OpenCV window.
    """
    color_patch = np.zeros((100, 100, 3), dtype=np.uint8)
    color_patch[:] = color  # Fill with the color
    cv2.imshow(window_name, color_patch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    base_img = cv2.imread("./samples/19.jpg") 
    new_barcode = cv2.imread("./samples/23.jpg")

    annotations = json.load(open("./annotations.json", "r"))
    src_region = annotations["_via_img_metadata"]["19.jpg134455"]["regions"][0]
    dst_region = annotations["_via_img_metadata"]["23.jpg161408"]["regions"][0]
    
    src_all_points_x = src_region["shape_attributes"]["all_points_x"]
    src_all_points_y = src_region["shape_attributes"]["all_points_y"]

    dst_all_points_x = dst_region["shape_attributes"]["all_points_x"]
    dst_all_points_y = dst_region["shape_attributes"]["all_points_y"]

    src_polygon = np.float32(list(zip(src_all_points_x, src_all_points_y)))
    print(src_polygon)

    dst_polygon = np.float32(list(zip(dst_all_points_x, dst_all_points_y)))
    print(dst_polygon)

    foreground_color, background_color = extract_colors(base_img, src_polygon)
    show_color(foreground_color, "Foreground")
    show_color(background_color, "Background")
    # seamless_replace(base_img, src_polygon, new_barcode, dst_polygon)

   

if __name__ == "__main__":
    main()