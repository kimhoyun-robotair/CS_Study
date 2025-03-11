import sys, os
# 현재 스크립트 파일의 절대 경로
script_dir = os.path.dirname(os.path.abspath(__file__))
# 스크립트 파일의 부모 디렉터리의 절대 경로
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
# 부모 디렉터리를 모듈 검색 경로에 추가
# 그냥 상대경로로 지정하니까 애가 인식을 못해서 절대 경로로 추가함
sys.path.append(parent_dir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

