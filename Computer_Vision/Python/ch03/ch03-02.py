# 가장 기본적인 opencv 이미지 읽어오기 및 출력
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")
h = cv.calcHist([img], [2], None, [256], [0, 256]) # 2번 채널인 R 채널에서 히스토그램 구하기
plt.plot(h, color="r", linewidth=2)
plt.show()
