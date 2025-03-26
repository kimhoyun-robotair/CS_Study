"""
글자를 사각형으로부터 좀 떨어뜨리고 화살표로 직사각형을 가리키도록 하기
"""
import cv2 as cv
import sys

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

cv.rectangle(img, (830,30), (1000,200), (0,0,255), 2) # 빨간색 사각형 그리기
cv.putText(img, "Starship", (110,20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) # 글자 쓰기
cv.arrowedLine(img, (110,20), (830, 30), (0, 0, 255), 2)

cv.imshow("Draw", img)
cv.waitKey(0)
cv.destroyAllWindows()

