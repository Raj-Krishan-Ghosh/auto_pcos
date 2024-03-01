import os
import torch
from PIL import Image, ImageFilter
import numpy as np
import config
from efficientnet_pytorch import EfficientNet
from torch import nn
import torch.nn.functional as F
import pickle
import pandas as pd
from utils import load_checkpoint

val_input_dir = "train_val/Dataset/val/"
val_inputs = sorted(os.listdir(val_input_dir))

val_preds = []

model = EfficientNet.from_pretrained("efficientnet-b7")
model._fc = nn.Linear(2560, 1)
model = model.to(config.DEVICE)
load_checkpoint(torch.load("best_model.pth.tar"), model)

model.eval()

for val_input in val_inputs:
    x = np.array(Image.open(os.path.join(val_input_dir + val_input))).astype(np.float32)
    
    transform = config.basic_transform
    x = transform(image=x)["image"]

    x = x.to(config.DEVICE)
    x = torch.unsqueeze(x, dim=0)
    with torch.no_grad():
        scores = model(x)
        prediction = torch.sigmoid(scores) > 0.5
        val_preds.append(int(prediction.item()))

df = pd.DataFrame({"S. No.": np.arange(1, len(val_inputs) + 1),"Image Path": val_inputs , "Predicted class label": val_preds})
df = pd.concat([pd.DataFrame([["S. No.", "Image Path (in ascending order)", "Predicted class label"]], columns=df.columns), df], ignore_index=True)

if(not os.path.isdir("train_val/val_predicted/")):
    os.makedirs("train_val/val_predicted/")

df.to_excel("train_val/val_predicted/val_predicted.xlsx", index=False, header=False)
print("Validation predictions finished.")

model.train()
