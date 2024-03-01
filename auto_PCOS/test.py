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

test_input_dir = "Dataset/PCOSGen-test/images/"
test_inputs = sorted(os.listdir(test_input_dir))

test_preds = []

model = EfficientNet.from_pretrained("efficientnet-b7")
model._fc = nn.Linear(2560, 1)
model = model.to(config.DEVICE)
load_checkpoint(torch.load("best_model.pth.tar"), model)

model.eval()

for test_input in test_inputs:
    x = np.array(Image.open(os.path.join(test_input_dir + test_input))).astype(np.float32)
    
    transform = config.basic_transform
    x = transform(image=x)["image"]

    x = x.to(config.DEVICE)
    x = torch.unsqueeze(x, dim=0)
    with torch.no_grad():
        scores = model(x)
        prediction = torch.sigmoid(scores) > 0.5
        test_preds.append(int(prediction.item()))

df = pd.DataFrame({"S. No.": np.arange(1, len(test_inputs) + 1),"Image Path": test_inputs , "Predicted class label": test_preds})
df = pd.concat([pd.DataFrame([["S. No.", "Image Path (in ascending order)", "Predicted class label"]], columns=df.columns), df], ignore_index=True)

if(not os.path.isdir("test/Predicted/")):
    os.makedirs("test/Predicted/")

df.to_excel("test/Predicted/Predicted.xlsx", index=False, header=False)
print("Test predictions finished.")

model.train()
