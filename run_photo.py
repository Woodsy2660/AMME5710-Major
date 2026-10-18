"""
Batch processing script for pool table ball detection.
Processes all images in the Table images folder and saves results.
"""

import os
import cv2
import numpy as np

import table_mask
import table_corners
import top_down
import ball_detection
import draw

def find_cloth_center(image_bgr):
    """
    Automatically finds a point on the cloth by looking for the dominant blue/green region.
    Returns (click_x, click_y) for a point likely to be on the table cloth.
    """
    height, width = image_bgr.shape[:2]

    # Convert to HSV and look for typical cloth colors (blue/green hues)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Pool table cloths are typically blue or green with high saturation
    hue = image_hsv[:, :, 0]
    saturation = image_hsv[:, :, 1]

    # Mask for blue-green hues (roughly 80-130 in OpenCV's 0-180 scale) with good saturation
    cloth_like = ((hue > 80) & (hue < 130) & (saturation > 50)).astype(np.uint8) * 255

    # Find the largest region
    contours, _ = cv2.findContours(cloth_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy

    # Fallback to image center
    return width // 2, height // 2

def detect(image_bgr):
    # Step 1: Find cloth center automatically
    click_x, click_y = find_cloth_center(image_bgr)

    # Step 2: Find cloth mask and table outline
    image_hsv, cloth_mask = table_mask.find_cloth_mask(image_bgr, click_x, click_y)
    table_contour, table_hull = table_mask.find_table_outline(cloth_mask, click_x, click_y)

    # Step 3: Find corners and orient them
    corners = table_corners.find_corners(table_hull)
    corners = table_corners.put_long_side_first(corners, table_contour)

    # Step 4: Create perspective transform
    to_map, to_photo = top_down.make_map_transform(corners)

    # Step 5: Detect balls
    balls = ball_detection.find_balls(image_hsv, cloth_mask, table_contour, table_hull, to_map, to_photo)
    
    return balls, table_hull, corners


def process_image(image_path, output_dir):
    """
    Processes a single image through the detection pipeline.
    Returns True on success, False on failure.
    """
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]

    print(f"Processing: {filename}")

    try:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"  Failed to load image")
            return False

        balls, table_hull, corners = detect(image_bgr)

        # Step 6: Draw results
        overlay = draw.draw_overlay(image_bgr, table_hull, corners, balls)
        top_down_map = draw.draw_map(balls)

        # Step 7: Combine overlay and map side by side
        map_height = top_down_map.shape[0]
        overlay_height = overlay.shape[0]

        # Resize map to match overlay height while keeping aspect ratio
        scale = overlay_height / map_height
        new_map_width = int(top_down_map.shape[1] * scale)
        map_resized = cv2.resize(top_down_map, (new_map_width, overlay_height))

        # Combine images horizontally
        combined = np.hstack([overlay, map_resized])

        # Save result
        output_path = os.path.join(output_dir, f"{name_without_ext}_result.jpg")
        cv2.imwrite(output_path, combined)

        print(f"  Found {len(balls)} balls")
        print(f"  Saved: {output_path}")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    """
    Main function to process all images in Table images folder.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "Table images")
    output_dir = os.path.join(script_dir, "results")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    success_count = 0
    for filename in sorted(image_files):
        image_path = os.path.join(input_dir, filename)
        if process_image(image_path, output_dir):
            success_count += 1

    print("-" * 50)
    print(f"Completed: {success_count}/{len(image_files)} images processed successfully")

if __name__ == "__main__":
    main()
