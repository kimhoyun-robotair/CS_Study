# 가장 기본적인 opencv 이미지 읽어오기 및 출력
import cv2 as cv
import sys

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")
t, bin_img = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print("오츄 알고리즘이 찾은 최적의 임계값 = ", t)

cv.imshow("Image", img)
cv.imshow("Binary Image", bin_img)
cv.waitKey(0)
cv.destroyAllWindows()
