# 마우스 왼쪽 버튼을 누르면 시작하고, 손을 떼면 끝나고 사각형이 그려짐
import cv2 as cv
import sys

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

def draw(event, x, y, flags, param): # callback 함수
    global ix, iy
    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼을 누름
        ix, iy = x, y
    elif event == cv.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼을 뗌
        cv.rectangle(img, (ix,iy), (x,y), (0,0,255), 2)

    cv.imshow("Drawing", img)

cv.namedWindow("Drawing")
cv.imshow("Drawing", img)

cv.setMouseCallback('Drawing', draw) # 콜백 함수 등록

while(True):
    if cv.waitKey(1) == ord('q'): # q가 입력되면 종료
        cv.destroyAllWindows()
        break
