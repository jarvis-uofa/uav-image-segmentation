import pandas as pd
import cv2
import numpy as np
import os

# --- Configuration ---
# Path to the folder containing your jpg images
image_folder = 'Data/rgb/'

# Path to your CSV file with bounding box coordinates
csv_path = 'rgb_interpolated_corners.csv'

# Directory where the output images will be saved
output_folder = 'output_frames/'

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# --- Load Data ---
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: CSV file not found at '{csv_path}'")
    exit()


# --- Helper Function for Drawing ---
def draw_translucent_polygon(image, points, color, alpha):
    """Draws a translucent filled polygon on an image."""
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color)
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


# --- Main Processing Loop ---
# Loop through each row of the DataFrame
for index, row in df.iterrows():
    frame_number = int(row['frame'])

    # Generate the zero-padded filename for the current frame
    image_filename = f"{frame_number:06d}.jpg"
    image_path = os.path.join(image_folder, image_filename)

    # Load the corresponding image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not load image for frame {frame_number} at '{image_path}'")
        continue

    # Organize the corner points into a NumPy array
    points = np.array([
        [row['top_left_x'], row['top_left_y']],
        [row['top_right_x'], row['top_right_y']],
        [row['bottom_right_x'], row['bottom_right_y']],
        [row['bottom_left_x'], row['bottom_left_y']]
    ], dtype=np.int32)

    # Define the color (BGR) and transparency for the box
    box_color = (0, 255, 0)  # Green
    transparency = 0.4

    # Draw the translucent polygon on the loaded image
    output_image = draw_translucent_polygon(image, points, box_color, transparency)

    # Save the processed image to the output folder
    output_path = os.path.join(output_folder, image_filename)
    cv2.imwrite(output_path, output_image)

    # Optional: Print progress
    if (index + 1) % 10 == 0:
        print(f"Processed and saved frame {index + 1}/{len(df)}")

print("\nProcessing complete.")
print(f"Output images are saved in the '{output_folder}' directory.")