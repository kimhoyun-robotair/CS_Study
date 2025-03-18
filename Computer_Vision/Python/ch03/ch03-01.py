# 가장 기본적인 opencv 이미지 읽어오기 및 출력
import cv2 as cv
import sys

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

cv.imshow("Image", img)
cv.imshow("Upper Left Half", img[0:img.shape[0]//2, 0:img.shape[1]//2,:])
cv.imshow("Center Half", img[img.shape[0]//4:3*img.shape[0]//4, img.shape[1]//4:3*img.shape[1]//4, :])

cv.imshow("R Channel", img[:,:,2])
cv.imshow("G Channel", img[:,:,1])
cv.imshow("B Channel", img[:,:,0])

cv.waitKey(0)
cv.destroyAllWindows()
# 이미지의 특정 부분을 잘라내어 출력하는 방법
