# 가장 기본적인 opencv 이미지 읽어오기 및 출력
import cv2 as cv
import sys

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")

print(img[0,0,0], img[0,0,0], img[0,0,2]) # pixel 조사
print(img[0,1,0], img[0,1,1], img[0,1,2]) # pixel 조사
# BGR 값이 출력

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()
