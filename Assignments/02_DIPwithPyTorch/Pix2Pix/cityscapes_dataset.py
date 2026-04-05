import torch
from torch.utils.data import Dataset
import cv2
import os

class CityscapesDataset(Dataset):
    def __init__(self, dir):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        self.dir = dir
        self.images = []
        for file_name in os.listdir(self.dir):
            self.images.append(os.path.join(self.dir, file_name))
        
    def __len__(self):
        # Return the total number of images
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.images[idx]
        img_color_semantic = cv2.imread(img_name)
        # Convert the image to a PyTorch tensor
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        image_rgb = image[:, :, :256]
        image_semantic = image[:, :, 256:]
        return image_rgb, image_semantic