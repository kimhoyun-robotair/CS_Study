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
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['b1'], network['b2'], network['b3']

    A1 = np.dot(x, W1) + B1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2, W3) + B3
    Y = softmax(A3)

    return Y

x, t = get_data()
network = init_network()

accurancy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accurancy_cnt += 1

print("Accurancy: " + str(float(accurancy_cnt) / len(x)))
