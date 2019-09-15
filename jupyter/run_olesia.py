import cv2
import numpy as np
import glob

images = [(file, cv2.imread(file)) for file in glob.glob("data/01_Pink_impTNT/01_pink_imptnt_01/*.jpg")]

for file, img in images:
    print(file[38:])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filt = cv2.bilateralFilter(img, 15, 5, 5)
    equ = cv2.equalizeHist(filt)
    th1 = cv2.adaptiveThreshold(filt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 195, 25)
    th2 = cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 195, 60)
    th = 255 - (255-th1) - (255-th2)
    kernel_co = np.ones((15, 15), np.uint8)
    kernel_dil = np.ones((20, 25), np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_co)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_co)
    dilation = cv2.erode(opening, kernel_dil, 1)
    mask = 255 - dilation
    edge = cv2.Canny(filt, 20, 255)
    edge_masked = edge * mask

    th_masked = th
    ids = np.argwhere(mask.flatten() == 0)
    th_masked = th_masked.flatten()
    th_masked[ids] = 255
    th_masked = th_masked.reshape(mask.shape[0], -1)
    cv2.imwrite("data/01_Pink_impTNT/01_pink_imptnt_01_1/{}".format(file[38:]), edge_masked)
