import sys, os
# 현재 스크립트 파일의 절대 경로
script_dir = os.path.dirname(os.path.abspath(__file__))
# 스크립트 파일의 부모 디렉터리의 절대 경로
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
# 부모 디렉터리를 모듈 검색 경로에 추가
# 그냥 상대경로로 지정하니까 애가 인식을 못해서 절대 경로로 추가함
sys.path.append(parent_dir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from sigmoid import sigmoid
from softmax import softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open(parent_dir + "/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))