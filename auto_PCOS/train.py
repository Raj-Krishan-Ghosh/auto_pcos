import os
import torch
import torch.nn.functional as F
import numpy as np
import config
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import UltrasoundData
from efficientnet_pytorch import EfficientNet
from utils import check_accuracy, load_checkpoint, save_checkpoint
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"


initial_seed = 0
torch.manual_seed(initial_seed)
torch.cuda.manual_seed_all(initial_seed)
np.random.seed(initial_seed)
random.seed(initial_seed)
torch.use_deterministic_algorithms(True)


def save_feature_vectors(model, loader, output_size=(1, 1), file="trainb7"):
    model.eval()
    images, labels = [], []

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)

        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=output_size)
        images.append(features.reshape(x.shape[0], -1).detach().cpu().numpy())
        labels.append(y.numpy())

    if(not os.path.isdir("train_val/data_features/")):
        os.makedirs("train_val/data_features/")
    np.save(f"train_val/data_features/X_{file}.npy", np.concatenate(images, axis=0))
    np.save(f"train_val/data_features/y_{file}.npy", np.concatenate(labels, axis=0))
    model.train()


def train_one_epoch(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(config.DEVICE)
        targets = targets.to(config.DEVICE).unsqueeze(1).float()

        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())


def main():
    model = EfficientNet.from_pretrained("efficientnet-b7")
    model._fc = nn.Linear(2560, 1)
    train_dataset = UltrasoundData(root="train_val/Dataset/train/", transform=config.basic_transform)
    val_dataset = UltrasoundData(root="train_val/Dataset/val/", transform=config.basic_transform)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    model = model.to(config.DEVICE)

    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    best_accuracy = 0.0

    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(train_loader, model, loss_fn, optimizer, scaler)
        accuracy_value = check_accuracy(val_loader, model, loss_fn)

        if accuracy_value > best_accuracy:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename="best_model.pth.tar")
            best_accuracy = accuracy_value

    if config.SAVE_MODEL:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)

if __name__ == "__main__":
    main()
