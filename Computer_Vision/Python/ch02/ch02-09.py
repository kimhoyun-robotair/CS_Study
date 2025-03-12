import cv2 as cv
import sys

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")
if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

BrushSiz = 5
LColor, RColor = (255, 0, 0), (0, 0, 255)  # BGR: 왼쪽은 파란색, 오른쪽은 빨간색

drawing_left = False
drawing_right = False # 일부 시스템에서는 마우스 입력이 제대로 되지 않아
# 글로벌 매개변수를 통해서 관리하기

def painting(event, x, y, flags, param):
    global drawing_left, drawing_right

    if event == cv.EVENT_LBUTTONDOWN:
        drawing_left = True
        cv.circle(img, (x, y), BrushSiz, LColor, -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        drawing_right = True
        cv.circle(img, (x, y), BrushSiz, RColor, -1)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing_left:
            cv.circle(img, (x, y), BrushSiz, LColor, -1)
        elif drawing_right:
            cv.circle(img, (x, y), BrushSiz, RColor, -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing_left = False
    elif event == cv.EVENT_RBUTTONUP:
        drawing_right = False

    cv.imshow("Painting", img)

cv.namedWindow("Painting")
cv.imshow("Painting", img)
cv.setMouseCallback("Painting", painting)

while True:
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
