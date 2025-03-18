import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")

t, bin_img = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
plt.imshow(bin_img, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

b = bin_img[bin_img.shape[0]//2:bin_img.shape[0], 0:bin_img.shape[0]//2+1]
plt.imshow(b, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

se = np.uint8([[0,0,1,0,0],
               [0,1,1,1,0],
               [1,1,1,1,1],
               [0,1,1,1,0],
               [0,0,1,0,0]])
b_dilation = cv.dilate(b, se, iterations=1)
plt.imshow(b_dilation, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

b_erosion = cv.erode(b_dilation, se, iterations=1)
plt.imshow(b_erosion, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

b_close = cv.erode(cv.dilate(b, se, iterations=1), se, iterations=1)
plt.imshow(b_close, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()
