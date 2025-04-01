import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

np.random.seed(777)
torch.manual_seed(777)

# 데이터 로드
xy = np.loadtxt("/home/kimhoyun/CS_Study/Automotive_AI/week04/diabetes.txt", delimiter=",", dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# 50:50 train/test 분할
total_size = x_data.shape[0]
indices = np.arange(total_size)
np.random.shuffle(indices)

split = total_size // 2
train_idx = indices[:split]
test_idx = indices[split:]

x_train = x_data[train_idx]
y_train = y_data[train_idx]
x_test = x_data[test_idx]
y_test = y_data[test_idx]

# Torch 텐서 변환
X = Variable(torch.from_numpy(x_train))
Y = Variable(torch.from_numpy(y_train))

# 모델 정의
linear = nn.Linear(8, 1, bias=True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear, sigmoid)

# 최적화
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습
for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = -(Y * torch.log(hypothesis) + (1 - Y) * torch.log(1 - hypothesis)).mean()
    cost.backward()
    optimizer.step()

    if step % 200 == 0:
        print(step, cost.item())

# ===== 테스트 정확도 평가 =====
# 테스트 데이터로 예측
x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.from_numpy(y_test)

# 예측 결과
with torch.no_grad():
    prediction = model(x_test_tensor)
    predicted_classes = (prediction >= 0.5).float()
    correct = (predicted_classes == y_test_tensor).sum()
    accuracy = correct / y_test_tensor.shape[0]
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

# ===== 트레인셋 정확도 평가 추가 =====
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)

with torch.no_grad():
    prediction_train = model(x_train_tensor)
    predicted_classes_train = (prediction_train >= 0.5).float()
    correct_train = (predicted_classes_train == y_train_tensor).sum()
    accuracy_train = correct_train / y_train_tensor.shape[0]
    print(f"Train Accuracy (50:50 split): {accuracy_train.item() * 100:.2f}%")
