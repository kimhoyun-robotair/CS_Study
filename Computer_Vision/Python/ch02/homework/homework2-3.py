# 각각 서로 다른 영상 2개를 읽어와 각각 img1, img2에 저장한 후, 두 영상을 서로 다른 윈도우에 출력하기
import cv2 as cv
import sys

img1 = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")
img2 = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship_gray.jpg")

if img1 is None or img2 is None:
    sys.exit("파일을 찾을 수 없습니다.")

cv.imshow("Image1", img1)
cv.imshow("Image2", img2)

cv.waitKey(0)
cv.destroyAllWindows()
