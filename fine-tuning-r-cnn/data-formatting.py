import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import pandas as pd
import numpy as np
import os 

from PIL import Image


class creepyBikeDataset(Dataset):
    def __init__(self, root_dr, transforms=None): # constructor statements, specifies where data is located too
        """
        Args:
            root_dir: Path to creepyBike dev folder, with ground truth
            transforms: Optional transforms to apply to images
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.samples = [] # stores (image_path, bounding_box) pairs

        # Parse video's image frame sequence in the dev training data folder
        self.parse_sequence()

    def parse_sequence(self): # parses only the sequence in the given directory
        rgb_folder = os.path.join(self.)