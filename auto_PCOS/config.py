import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 1  # 4
BATCH_SIZE = 4  # 64
PIN_MEMORY = True
LOAD_MODEL = True  # True
SAVE_MODEL = True
CHECKPOINT_FILE = "b7.pth.tar"
WEIGHT_DECAY = 1e-4  # 1e-4
LEARNING_RATE = 1e-4  # 1e-4
NUM_EPOCHS = 10  # 1

basic_transform = A.Compose(
    [
        A.Resize(height=256, width=256),
        ToTensorV2(),
    ]
)
