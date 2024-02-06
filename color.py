import cv2 as cv
import numpy as np

bgr = cv.imread("color/test/00000000_(3).jpg")
lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)[:, :, 1:]
gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
gray = np.expand_dims(gray, axis=2)

cat = np.concatenate((gray, lab), axis=2)

print(cat[0][0][0], cat[0][0][1], cat[0][0][2])
out = cv.cvtColor(cat, cv.COLOR_LAB2BGR)
print(out[0][0][0], out[0][0][1], out[0][0][2])
cv.imwrite("output.jpg", out)
