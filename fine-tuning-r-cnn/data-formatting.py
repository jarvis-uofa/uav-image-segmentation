import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import pandas as pd
import numpy as np
import os 

from PIL import Image


class creepyBikeDataset(Dataset):
    def __init__(self, data_dir, transforms=None): # constructor statements, specifies where data is located too
        """
        Args:
            root_dir: Path to creepyBike dev folder, with ground truth
            transforms: Optional transforms to apply to images
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.samples = [] # stores (image_path, bounding_box) pairs

        # Parse video's image frame sequence in the dev training data folder
        self.parse_sequence()

    def parse_sequence(self): # parses only the sequence in the given directory
        rgb_folder = os.path.join(self.data_dir, 'rgb') # path to rgb folder within data folder
        ground_truth_csv = os.path.join(self.data_dir, 'ground-truth.csv')

        ground_truth_dataframe = pd.read_csv(
            ground_truth_csv, 
            names=['frame', 'bottom_left_x', 'bottom_left_y', 'top_right_x', 'top_right_y']
            )
        sorted_image_files = sorted(os.listdir(rgb_folder)) # sort from 000001.png to 0003556.png

        for rowIndex, row in ground_truth_dataframe.itterows():
            frame_index = int(row['frame']) - 1 # frame index in csv starts from 1, but list index starts from 0

            image_path = 
