"""
이미지를 0.1, 0.2 ... 0.9, 1.0배로 축소한 영상 10개를 서로 다른 디스플레이에 출력
"""
import cv2 as cv
import sys

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

img_list = []
for i in range(10):
    img_small = cv.resize(img, dsize=(0,0), fx=0.1*(i+1), fy=0.1*(i+1))
    img_list.append(img_small)

cv.imshow("Image", img)
cv.imshow("Image 10%", img_list[0])
cv.imshow("Image 20%", img_list[1])
cv.imshow("Image 30%", img_list[2])
cv.imshow("Image 40%", img_list[3])
cv.imshow("Image 50%", img_list[4])
cv.imshow("Image 60%", img_list[5])
cv.imshow("Image 70%", img_list[6])
cv.imshow("Image 80%", img_list[7])
cv.imshow("Image 90%", img_list[8])
cv.imshow("Image 100%", img_list[9])

cv.waitKey(0)
cv.destroyAllWindows()
