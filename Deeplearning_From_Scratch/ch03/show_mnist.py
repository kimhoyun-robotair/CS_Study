import sys, os
# 현재 스크립트 파일의 절대 경로
script_dir = os.path.dirname(os.path.abspath(__file__))
# 스크립트 파일의 부모 디렉터리의 절대 경로
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
# 부모 디렉터리를 모듈 검색 경로에 추가
# 그냥 상대경로로 지정하니까 애가 인식을 못해서 절대 경로로 추가함
sys.path.append(parent_dir)
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label) # 5

print(img.shape) # (784,)
img= img.reshape(28, 28) # 원래 이미지 모양으로 변형
print(img.shape) # (28, 28)

img_show(img)
