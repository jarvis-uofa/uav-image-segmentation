import pandas as pd
import numpy as np
import csv

# --- Step 2: Define the interpolation function ---
def linear_interpolate(start, end, t, total_t):
    """
    Calculates the linearly interpolated value.
    - start: The starting value (e.g., initial corner's x-coordinate).
    - end: The ending value (e.g., final corner's x-coordinate).
    - t: The number of frames elapsed since the start frame.
    - total_t: The total number of frames in the interval (e.g., 10).
    """
    fraction = t / total_t
    return start + (end - start) * fraction
# --- Step 3: Load data and calculate initial corners ---
# Define the column names you expect
column_names = ['x', 'y', 'width', 'height']
# Read the CSV, specifying there's no header and providing the correct names
df_ground_truth = pd.read_csv('rgb.csv')
# Convert all relevant columns to a numeric type (e.g., integers)
# This is the crucial step to fix the TypeError
for col in column_names:
    df_ground_truth[col] = pd.to_numeric(df_ground_truth[col])

# Assume ground truth is every 10 frames, starting at frame 1
df_ground_truth['frame'] = (np.arange(len(df_ground_truth)) * 10) + 1

# Calculate the four corner coordinates for each ground truth frame
df_ground_truth['top_left_x'] = df_ground_truth['x']
df_ground_truth['top_left_y'] = df_ground_truth['y']
df_ground_truth['top_right_x'] = df_ground_truth['x'] + df_ground_truth['width']
df_ground_truth['top_right_y'] = df_ground_truth['y']
df_ground_truth['bottom_left_x'] = df_ground_truth['x']
df_ground_truth['bottom_left_y'] = df_ground_truth['y'] + df_ground_truth['height']
df_ground_truth['bottom_right_x'] = df_ground_truth['x'] + df_ground_truth['width']
df_ground_truth['bottom_right_y'] = df_ground_truth['y'] + df_ground_truth['height']

# --- Step 4: Perform interpolation between ground truth frames ---
interpolated_rows = []
corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

# Loop through pairs of consecutive ground truth frames
for i in range(len(df_ground_truth) - 1):
    start_frame = df_ground_truth.iloc[i]
    end_frame = df_ground_truth.iloc[i+1]
    
    total_frames_in_interval = end_frame['frame'] - start_frame['frame']

    # Generate interpolated frames for the interval
    for t in range(total_frames_in_interval):
        frame_number = start_frame['frame'] + t
        interpolated_row = {'frame': frame_number}
        
        # Interpolate each corner's x and y coordinates
        for corner in corners:
            start_x = start_frame[f'{corner}_x']
            end_x = end_frame[f'{corner}_x']
            interpolated_row[f'{corner}_x'] = linear_interpolate(start_x, end_x, t, total_frames_in_interval)
            
            start_y = start_frame[f'{corner}_y']
            end_y = end_frame[f'{corner}_y']
            interpolated_row[f'{corner}_y'] = linear_interpolate(start_y, end_y, t, total_frames_in_interval)
            
        interpolated_rows.append(interpolated_row)

# Add the very last ground truth frame, which the loop doesn't cover
last_frame_data = df_ground_truth.iloc[-1]
final_row = {'frame': last_frame_data['frame']}
for corner in corners:
    final_row[f'{corner}_x'] = last_frame_data[f'{corner}_x']
    final_row[f'{corner}_y'] = last_frame_data[f'{corner}_y']
interpolated_rows.append(final_row)

# --- Step 5: Finalize and save the interpolated data ---
df_interpolated = pd.DataFrame(interpolated_rows)

# Round all corner coordinates to the nearest integer
for col in df_interpolated.columns:
    if col != 'frame':
        df_interpolated[col] = df_interpolated[col].round().astype(int)

# Save the final data to a new CSV file
output_interpolated_filename = 'rgb_interpolated_corners.csv'
df_interpolated.to_csv(output_interpolated_filename, index=False)

print(f"Interpolation complete. Data saved to '{output_interpolated_filename}'.")
