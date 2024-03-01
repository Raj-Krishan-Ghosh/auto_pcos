import os
import re
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class UltrasoundData(Dataset):
    def __init__(self, root, transform=None):
        self.images = os.listdir(root)
        self.images.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        self.root = root
        self.transform = transform
        self.df = pd.read_excel('Dataset/PCOSGen-train/class_label.xlsx')

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        file = self.images[index]
        img = np.array(Image.open(os.path.join(self.root, file))).astype(np.float32)
        try:
            label = self.df.loc[self.df['imagePath'] == file]
            label = label["Healthy"].item()
        except:
            label = -1

        if "test" in self.root:
            label = -1
        
        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, label
