
import json
import numpy as np
import cv2


def blur_kernel_size(bounding_box):
    side_lengths = [
        np.linalg.norm(bounding_box[i] - bounding_box[(i + 1) % len(bounding_box)])
        for i in range(len(bounding_box))
    ]
    
    # Compute the average side length
    average_dim = int(np.mean(side_lengths))
    kernel_size = max(1, int(average_dim * 0.01) | 1)  # At least 1 and odd
    print("Kernel size: ", kernel_size)

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
    kernel_size = max(1, int(avg_side_length * 0.01))
    print("Dilation size: ", kernel_size)
    return np.ones((kernel_size, kernel_size), np.uint8)


def create_expanded_mask(polygon, img):
    # Create a mask for the barcode area
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(int), 255)
    
    # Expand mask a little
    expanded_mask = np.zeros_like(mask)
    cv2.dilate(mask, dilation_kernel(polygon), dst=expanded_mask)
    
    # Apply blurring for smoother replacement 
    kernel_size = blur_kernel_size(polygon)
    return cv2.GaussianBlur(expanded_mask, (kernel_size, kernel_size), 0)


def get_background_and_foreground(image, polygon):
    """
    Extracts the dominant foreground and background masks from a region.
    
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
    if cv2.countNonZero(255 * (foreground_mask & border_mask)) > cv2.countNonZero(255 * (background_mask & border_mask)):
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
    foreground_color = np.mean(image[foreground_mask], axis=0)  # Average color of the foreground
    background_color = np.mean(image[background_mask], axis=0)  # Average color of the background

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

def correct_colors(base_img, src_polygon, new_barcode, dst_polygon):
    background_color, foreground_color = extract_colors(base_img, src_polygon)
    return recolor_barcode(new_barcode, dst_polygon, background_color, foreground_color)

def seamless_replace(base_img, src_polygon, new_barcode, dst_polygon):
    new_barcode = correct_colors(base_img, src_polygon, new_barcode, dst_polygon)

    M = cv2.getPerspectiveTransform(dst_polygon, src_polygon)
    warped_barcode  = cv2.warpPerspective(new_barcode, M, (base_img.shape[1], base_img.shape[0]))

    # Seamless cloning
    mask = create_expanded_mask(src_polygon, base_img)

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

    ids = [
		"1.jpg74093",
		"2.jpg84623",
		"3.jpg166066",
		"4.jpg179426",
		"5.jpg133037",
		"6.jpg79954",
		"7.jpg126295",
		"8.png848842",
		"9.png621283",
		"10.jpg146372",
		"11.jpg129662",
		"12.jpg129054",
		"13.jpg73923",
		"14.jpg141047",
		"16.jpg103336",
		"17.jpg64551",
		"18.jpg54553",
		"19.jpg134455",
		"20.jpg119620",
		"21.jpg103242",
		"22.jpg56428",
		"23.jpg161408",
		"24.jpg81321",
		"25.jpg49014",
		"26.jpg81972",
		"27.jpg171610",
		"28.jpg199540",
		"29.jpg79065",
		"30.jpg53070",
		"31.jpg97889",
		"32.jpg75349"]
    
    names = [
		"1.jpg",
		"2.jpg",
		"3.jpg",
		"4.jpg",
		"5.jpg",
		"6.jpg",
		"7.jpg",
		"8.png",
		"9.png",
		"10.jpg",
		"11.jpg",
		"12.jpg",
		"13.jpg",
		"14.jpg",
		"16.jpg",
		"17.jpg",
		"18.jpg",
		"19.jpg",
		"20.jpg",
		"21.jpg",
		"22.jpg",
		"23.jpg",
		"24.jpg",
		"25.jpg",
		"26.jpg",
		"27.jpg",
		"28.jpg",
		"29.jpg",
		"30.jpg",
		"31.jpg",
		"32.jpg"]
    for name, id in zip(names, ids):
            
        base_img = cv2.imread("./samples/" + name) 
        new_barcode = cv2.imread("./image.png")

        annotations = json.load(open("./annotations.json", "r"))
        src_region = annotations["_via_img_metadata"][id]["regions"][0]
        
        src_all_points_x = src_region["shape_attributes"]["all_points_x"]
        src_all_points_y = src_region["shape_attributes"]["all_points_y"]

        dst_all_points_x = [145,266,264,145]
        dst_all_points_y = [165,165,44,45]


        src_polygon = np.float32(list(zip(src_all_points_x, src_all_points_y)))
        
        if src_polygon.shape[0] > 4:
            continue

        print(name)

        dst_polygon = np.float32(list(zip(dst_all_points_x, dst_all_points_y)))

        # foreground_color, background_color = extract_colors(base_img, src_polygon)
        # show_color(foreground_color, "Foreground")
        # show_color(background_color, "Background")
        # new_barcode = correct_colors(base_img, src_polygon, new_barcode, dst_polygon)
        # cv2.imshow("New barcode recolored", new_barcode)
        # cv2.waitKey()
        seamless_replace(base_img, src_polygon, new_barcode, dst_polygon)

   

if __name__ == "__main__":
    main()