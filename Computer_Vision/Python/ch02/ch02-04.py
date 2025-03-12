# 웹 캠에서 비디오 읽어오기
import cv2 as cv
import sys

cap = cv.VideoCapture(0, cv.CAP_DSHOW) # 카메라와 연결 시도

if not cap.isOpened():
    sys.exit("카메라를 열 수 없습니다.")

while True:
    ret, frame = cap.read() # 비디오를 구성하는 프레임 획득
    if not ret:
        print("비디오 읽기 오류")
        break

    cv.imshow("Video display", frame) # 프레임 출력

    key = cv.waitKey(1) # 1ms 동안 키 입력 대기
    if key == ord('q'):
        break # q를 누르면 종료

cap.release() # 카메라와의 연결 해제
cv.destroyAllWindows() # 창 닫기
