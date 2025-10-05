import os

def remove_leading_zeros(folder_path):
    """
    Remove leading zeros from all image filenames in the specified folder.

    Args:
        folder_path (str): Path to the folder containing image files.
    """
    files = os.listdir(folder_path)

    for filename in files:
        name, ext = os.path.splitext(filename)

        # Convert to int, remove leading zeros, then convert back to string
        new_filename = str(int(name)) + ext
        
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)

        os.rename(old_file_path, new_file_path)

if __name__ == "__main__":
    rgb_folder_path = "data/rgb"

    remove_leading_zeros(rgb_folder_path)
    
    print("Leading zeros removed from filenames in the folder.")
