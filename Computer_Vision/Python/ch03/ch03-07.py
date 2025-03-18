import cv2 as cv
import numpy as np

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")
img = cv.resize(img, dsize=(0,0), fx = 0.4, fy = 0.4)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.putText(gray, "Original", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv.imshow("Original", gray)

smooth = np.hstack((cv.GaussianBlur(gray, (5,5), 0),
                    cv.GaussianBlur(gray, (9,9), 0),
                    cv.GaussianBlur(gray, (15,15), 0.0))) # gaussian blurring을 적용
cv.imshow("Gaussian", smooth)

femboss = np.array([[-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0]])

gray16 = np.int16(gray)
emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss)+128, 0, 255))
emboss_bd = np.uint8(cv.filter2D(gray16, -1, femboss)+128)
emboss_worse = cv.filter2D(gray, -1, femboss)

cv.imshow('Emboss', emboss)
cv.imshow("Emboss_bd", emboss_bd)
cv.imshow("Emboss_worse", emboss_worse)

cv.waitKey(0)
cv.destroyAllWindows()
