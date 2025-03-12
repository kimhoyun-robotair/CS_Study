# 그레이 스케일로 변환하고 사이즈 조정하기
import cv2 as cv
import sys

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BGR 컬러 영상을 그레이 스케일로 변환
gray_small = cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5) # 영상을 절반으로 축소

cv.imwrite("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship_gray.jpg", gray)
cv.imwrite("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/startship_gray_small.jpg", gray_small)
# 영상을 파일에 저장

cv.imshow("Image", img)
cv.imshow("Gray Image", gray)
cv.imshow("Gray Image Small", gray_small)

cv.waitKey(0)
cv.destroyAllWindows()
