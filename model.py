import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import cv2 as cv
import numpy as np

from config import IMG_SIZE, LEARNING_RATE, EPOCHS, BATCH_SIZE

class LandscapeDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        color = cv.imread(os.path.join(self.image_dir, self.image_files[idx]))
        lab = cv.cvtColor(color, cv.COLOR_BGR2LAB)
        ab_trans = np.transpose(lab, (2, 0, 1))[1:]
        ab_tensor = torch.tensor(ab_trans).float()
        ab_tensor_flat = torch.flatten(ab_tensor)

        grey = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
        grey_exp = np.expand_dims(grey, axis=0)
        grey_tensor = torch.tensor(grey_exp).float()

        return grey_tensor, ab_tensor_flat


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        img_pixels = IMG_SIZE * IMG_SIZE
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(img_pixels, int(img_pixels / 32)),
            nn.ReLU(),
            nn.Linear(int(img_pixels / 32), int(img_pixels / 128)),
            nn.ReLU(),
            nn.Linear(int(img_pixels / 128), int(img_pixels / 1024)),
            nn.ReLU(),
            nn.Linear(int(img_pixels / 1024), int(img_pixels / 8)),
            nn.ReLU(),
            nn.Linear(int(img_pixels / 8), img_pixels * 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
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

        if batch % 10 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    train_dataset = LandscapeDataset("images/train")
    test_dataset = LandscapeDataset("images/test")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)

    return model

if __name__ == '__main__':
    model = create_model()
    torch.save(model.state_dict(), "model.pth")
