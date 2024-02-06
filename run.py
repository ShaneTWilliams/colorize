import sys

import torch
from torch import nn
import cv2 as cv
import numpy as np

from config import IMG_SIZE
from model import NeuralNetwork

if __name__ == '__main__':
    model = NeuralNetwork()
    model.load_state_dict(torch.load('model.pth'))
    input_raw = cv.imread(sys.argv[1])
    input_gray = cv.cvtColor(input_raw, cv.COLOR_BGR2GRAY)
    input_exp1 = np.expand_dims(input_gray, axis=0)
    input_exp2 = np.expand_dims(input_exp1, axis=0)
    input_tensor = torch.tensor(input_exp2).float()

    logits = model(input_tensor)

    output_raw = torch.unflatten(logits, 1, (2, IMG_SIZE, IMG_SIZE))
    output_np = output_raw.detach().numpy()[0]
    output_cat = np.concatenate((input_exp1, output_np), axis=0)
    output_trans = output_cat.transpose(1, 2, 0)
    output_int = output_trans.astype(np.uint8)
    output_lab = cv.cvtColor(output_int, cv.COLOR_LAB2BGR)

    print(cv.imwrite("output.jpg", output_lab))
