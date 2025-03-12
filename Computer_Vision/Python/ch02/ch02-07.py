import cv2 as cv
import sys

img = cv.imread("/home/kimhoyun/CS_Study/Computer_Vision/Python/img/starship.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

def draw(event, x, y, flags, param): # callback 함수
    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼을 누름
        cv.rectangle(img,(x,y), (x+200, y+200), (0,0,255), 2)
    elif event == cv.EVENT_RBUTTONDOWN: # 오른쪽 버튼을 클릭했을 때
        cv.rectangle(img,(x,y), (x+100, y+100), (255,0,0), 2)
    cv.imshow("Drawing", img)

cv.namedWindow("Drawing")
cv.imshow("Drawing", img)
cv.setMouseCallback("Drawing", draw) # 콜백 함수 등록

while(True):
    if cv.waitKey(1) == ord('q'): # 마우스 이벤트가 언제 발생할지 모르니 무한반복
        # q가 입력되면 종료
        cv.destroyAllWindows()
        break
