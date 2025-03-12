# 비디오에서 영상을 불러와서 저장하기
import cv2 as cv
import sys
import numpy as np

cap = cv.VideoCapture(0, cv.CAP_DSHOW) # 카메라와 연결 시도

if not cap.isOpened():
    sys.exit("카메라를 열 수 없습니다.")

frames = []
while True:
    ret, frame = cap.read() # 비디오를 구성하는 프레임 획득
    if not ret:
        print("비디오 읽기 오류")
        break

    cv.imshow("Video display", frame) # 프레임 출력

    key = cv.waitKey(1) # 1ms 동안 키 입력 대기
    if key == ord('c'): # c를 누르면 프레임을 리스트에 추가
        frames.append(frame)
    elif key == ord('q'): # q를 누르면 루프를 빠져나감
        break

cap.release() # 카메라와의 연결 해제
cv.destroyAllWindows() # 창 닫기

if len(frames) == 0:
    imgs = frames[0]
    for i in range(1, min(3, len(frames))):
        img = np.hstack((imgs, frames[i]))

    cv.imshow("collected images", img)

    cv.waitKey(0)
    cv.destroyAllWindows()
