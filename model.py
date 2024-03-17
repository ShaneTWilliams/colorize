import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import cv2 as cv
import numpy as np

from config import LEARNING_RATE, EPOCHS, BATCH_SIZE, TRAIN_RATIO

class LandscapeDataset(Dataset):
    def __init__(self, image_dir, device, train=True):
        self.device = device
        all_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
        num_files = len(all_files)
        cutoff = int(num_files * TRAIN_RATIO)
        self.image_files = all_files[:cutoff] if train else all_files[cutoff:]
        print(f"{'Train:' if train else 'Test: '} Found {len(self.image_files)} images at {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        color = cv.imread(self.image_files[idx])
        lab = cv.cvtColor(color, cv.COLOR_BGR2LAB)
        ab_trans = np.transpose(lab, (2, 0, 1))[1:]
        ab_tensor = torch.tensor(ab_trans).float()

        grey = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
        grey_exp = np.expand_dims(grey, axis=0)
        grey_tensor = torch.tensor(grey_exp).float()

        return grey_tensor.to(self.device), ab_tensor.to(self.device)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),

            nn.Softmax(dim=1),
            nn.Conv2d(256, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False),
            nn.Upsample(scale_factor=4, mode='bilinear'),
        )

    def forward(self, x):
        x = (x - 128) / 128.
        logits = self.layers(x)
        logits = (logits * 128.) + 128
        return logits


def train_loop(dataloader, model, loss_fn, optimizer, epoch, start_time):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), batch * BATCH_SIZE + len(X)
        elapsed_seconds = int(time.time() - start_time)
        print(f"{elapsed_seconds:05}: e{epoch} loss={loss:>7.2f} \t[{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def create_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_dataset = LandscapeDataset("lhq_256", device, train=True)
    test_dataset = LandscapeDataset("lhq_256", device, train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()
    for t in range(EPOCHS):
        train_loop(train_loader, model, loss_fn, optimizer, t+1, start_time)
        test_loop(test_loader, model, loss_fn)

    return model

if __name__ == '__main__':
    model = create_model()
    torch.save(model.state_dict(), "model.pth")
