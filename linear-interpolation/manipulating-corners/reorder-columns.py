import pandas as pd

# Read your CSV file
df = pd.read_csv('rgb_inter_mod.csv')

# Reorder columns
df_reordered = df[['frame', 'bottom_left_x', 'bottom_left_y', 'top_right_x', 'top_right_y']]

# Save to new file
df_reordered.to_csv('reordered_output.csv', index=False)