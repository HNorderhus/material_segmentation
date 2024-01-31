import glob
import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch


class DataLoaderSegmentation(Dataset):
    def __init__(self, folder_path, transform=None):
        self.img_files = sorted(glob.glob(os.path.join(folder_path, 'Images', '*.*')))
        self.mask_files = sorted(glob.glob(os.path.join(folder_path, 'Labels_grayscale', '*.*')))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # After transformations, ensure mask tensor is of type Long
        # The squeeze operation is used to remove a singleton dimension if it exists,
        # then convert to Long type
        mask = mask.squeeze().long()

        return image, mask



    def __len__(self):
        return len(self.img_files)