import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter

base_folder = "Dataset/PCOSGen-train/"
train_folder = "train_val/Dataset/train/"
if(not os.path.isdir(train_folder)):
    os.makedirs(train_folder)
val_folder = "train_val/Dataset/val/"
if(not os.path.isdir(val_folder)):
    os.makedirs(val_folder)

class_label_df = pd.read_excel('Dataset/PCOSGen-train/class_label.xlsx')

# ensuring equal class distribution in training set and validation set
class_0_label_df = class_label_df.loc[class_label_df['Healthy'] == 0]
class_0_label_df = class_0_label_df.sample(frac = 1, random_state=0)  # shuffling
class_1_label_df = class_label_df.loc[class_label_df['Healthy'] == 1]
class_1_label_df = class_1_label_df.sample(frac = 1, random_state=1)  # shuffling

# splitting into training and testing dataset (shuffling is already done to ensure random train and test allocation)
train_class_0_label_df = class_0_label_df[:int(class_0_label_df.shape[0] * 0.8)]
val_class_0_label_df = class_0_label_df[int(class_0_label_df.shape[0] * 0.8):]
train_class_1_label_df = class_1_label_df[:int(class_1_label_df.shape[0] * 0.8)]
val_class_1_label_df = class_1_label_df[int(class_1_label_df.shape[0] * 0.8):]

train_class_label_df = pd.concat([train_class_0_label_df, train_class_1_label_df])
val_class_label_df = pd.concat([val_class_0_label_df, val_class_1_label_df])

train_list = np.array(train_class_label_df["imagePath"])
val_list = np.array(val_class_label_df["imagePath"])

for file_path in train_list:
    img = Image.open(base_folder + "images/" + file_path)
    img.save(train_folder + file_path)

for file_path in val_list:
    img = Image.open(base_folder + "images/" + file_path)
    img.save(val_folder + file_path)
