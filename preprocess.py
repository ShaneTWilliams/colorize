import os

import cv2 as cv
import numpy as np

from config import IMG_SIZE

def preprocess_image(path):
    img = cv.imread(path)
    size = min(*img.shape[:2])
    y_start = (img.shape[0] - size) // 2
    x_start = (img.shape[1] - size) // 2
    crop_img = img[y_start:y_start+size, x_start:x_start+size]
    downscaled_img = cv.resize(crop_img, (IMG_SIZE, IMG_SIZE))
    grey_img = cv.cvtColor(downscaled_img, cv.COLOR_BGR2GRAY)
    return grey_img, downscaled_img

if __name__ == '__main__':
    for subdir in ["test", "train"]:
        i = 1
        os.makedirs(os.path.join("grey", subdir), exist_ok=True)
        os.makedirs(os.path.join("color", subdir), exist_ok=True)
        for file in os.listdir(os.path.join("input", subdir)):
            print(f"{i:04}: Processing {subdir}/{file}")
            grey_img, downscaled_img = preprocess_image(os.path.join("input", subdir, file))
            cv.imwrite(os.path.join("grey", subdir, file), grey_img)
            cv.imwrite(os.path.join("color", subdir, file), downscaled_img)
            i += 1
