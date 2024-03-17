import sys
import os

import torch
import cv2 as cv
import numpy as np

from model import NeuralNetwork
from config import IMG_SIZE

SATURATION_COEFF = 1.5
NUM_IMAGES = 1000
OUTPUT_DIR = "output"
VIDEO_INPUT_DIR = "videos"
NUM_IMAGES = 90000

def pass_image(model, input_gray):
    input_exp1 = np.expand_dims(input_gray, axis=0)
    input_exp2 = np.expand_dims(input_exp1, axis=0)
    input_tensor = torch.tensor(input_exp2).to("cuda").float()
    logits = model(input_tensor)

    output_np = logits.detach().to("cpu").numpy()[0]
    output_cat = np.concatenate((input_exp1, output_np), axis=0)
    output_trans = output_cat.transpose(1, 2, 0)
    output_int = output_trans.astype(np.uint8)
    output_bgr = cv.cvtColor(output_int, cv.COLOR_LAB2BGR)

    hsv = cv.cvtColor(output_bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    (h, s, v) = cv.split(hsv)
    s = s * SATURATION_COEFF
    s = np.clip(s, 0, 255)
    hsv_sat = cv.merge([h,s,v])
    output_bgr_sat = cv.cvtColor(hsv_sat.astype(np.uint8), cv.COLOR_HSV2BGR)

    return output_bgr, output_bgr_sat

def main(action, model_name):
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    image_dir = os.path.join(output_dir, "images")
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    model = NeuralNetwork()
    model.load_state_dict(torch.load(f"{model_name}.pth"))
    model.to("cuda")

    if action == "image":
        image = 0
        while True:
            if image % 100 == 0:
                print(f"Processing image {image}")
            image_path = f"lhq_256/{image:07}.png"
            input_raw = cv.imread(image_path)
            input_gray = cv.cvtColor(input_raw, cv.COLOR_BGR2GRAY)
            output_bgr, output_bgr_sat = pass_image(model, input_gray)
            input_gray = cv.cvtColor(input_gray, cv.COLOR_GRAY2BGR)

            muxed = np.zeros((IMG_SIZE*2, IMG_SIZE*2, 3), dtype=np.uint8)
            muxed[:IMG_SIZE, :IMG_SIZE] = input_raw
            muxed[:IMG_SIZE, IMG_SIZE:] = input_gray
            muxed[IMG_SIZE:, :IMG_SIZE] = output_bgr
            muxed[IMG_SIZE:, IMG_SIZE:] = output_bgr_sat

            cv.imshow("Image Result", muxed)
            key = cv.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            elif key == 81:
                image -= 1 if image > 0 else 0
            elif key == 83:
                image += 1 if image < NUM_IMAGES else 0
            elif key == ord('s'):
                output_path = os.path.join(image_dir, f"{image:07}.png")
                cv.imwrite(output_path, muxed)
                print(f"Saved image {image_path} to {output_path}")

    elif action == "video":
        video = "mountains"
        input_video_path = os.path.join(VIDEO_INPUT_DIR, f"{video}.mp4")
        cap = cv.VideoCapture(input_video_path)
        fourcc = cv.VideoWriter_fourcc(*'MP4V')
        output_video_path = os.path.join(video_dir, f"{video}.mp4")
        out = cv.VideoWriter(
            output_video_path, fourcc, 20.0, (IMG_SIZE * 2, IMG_SIZE * 2))

        while cap.isOpened():
            ret, frame = cap.read()

            # if frame is read correctly ret is True
            if not ret:
                break

            size = min(*frame.shape[:2])
            y_start = (frame.shape[0] - size) // 2
            x_start = (frame.shape[1] - size) // 2
            crop = frame[y_start:y_start+size, x_start:x_start+size]

            color256 = cv.resize(crop, (IMG_SIZE, IMG_SIZE))
            gray256 = cv.cvtColor(color256, cv.COLOR_BGR2GRAY)
            colorized, colorized_sat = pass_image(model, gray256)
            gray256 = cv.cvtColor(gray256, cv.COLOR_GRAY2BGR)

            muxed = np.zeros((IMG_SIZE*2, IMG_SIZE*2, 3), dtype=np.uint8)
            muxed[:IMG_SIZE, :IMG_SIZE] = color256
            muxed[:IMG_SIZE, IMG_SIZE:] = gray256
            muxed[IMG_SIZE:, :IMG_SIZE] = colorized
            muxed[IMG_SIZE:, IMG_SIZE:] = colorized_sat

            out.write(muxed)

            cv.imshow("Video Result", muxed)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()

    else:
        print("Invalid action")

if __name__ == '__main__':
    try:
        action = sys.argv[1]
        model_name = sys.argv[2]
    except IndexError:
        print("Usage: python run.py <action> <model_name>")
        sys.exit(1)

    main(action, model_name)
