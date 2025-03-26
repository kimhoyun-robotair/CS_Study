import cv2 as cv
import sys

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")
if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

BrushSiz = 5
LColor, RColor = (255, 0, 0), (0, 0, 255)  # BGR: 왼쪽은 파란색, 오른쪽은 빨간색

drawing_left = False
drawing_right = False  # 마우스 이벤트를 전역 변수로 관리

def painting(event, x, y, flags, param):
    global drawing_left, drawing_right, BrushSiz

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
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('+'):
        BrushSiz += 1
        print("Brush size increased to", BrushSiz)
    elif key == ord('-'):
        BrushSiz = max(1, BrushSiz - 1)  # 붓의 크기가 1 이하로 내려가지 않도록 함
        print("Brush size decreased to", BrushSiz)

cv.destroyAllWindows()
