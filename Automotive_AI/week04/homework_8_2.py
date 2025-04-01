import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

# 시드 고정
np.random.seed(777)
torch.manual_seed(777)

# 데이터 로드
xy = np.loadtxt("/home/kimhoyun/CS_Study/Automotive_AI/week04/diabetes.txt",
                delimiter=",", dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# 80:20 train/test 분할
total_size = x_data.shape[0]
indices = np.arange(total_size)
np.random.shuffle(indices)

split = int(total_size * 0.8)
train_idx = indices[:split]
test_idx = indices[split:]

x_train = x_data[train_idx]
y_train = y_data[train_idx]
x_test = x_data[test_idx]
y_test = y_data[test_idx]

# Torch 텐서로 변환
X = Variable(torch.from_numpy(x_train))
Y = Variable(torch.from_numpy(y_train))

# 모델 정의
model = nn.Sequential(
    nn.Linear(8, 1, bias=True),
    nn.Sigmoid()
)

# 최적화 함수
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습 루프
for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = -(Y * torch.log(hypothesis) + (1 - Y) * torch.log(1 - hypothesis)).mean()
    cost.backward()
    optimizer.step()

    if step % 200 == 0:
        print(step, cost.item())

# ===== 테스트셋 정확도 평가 =====
x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.from_numpy(y_test)

with torch.no_grad():
    prediction_test = model(x_test_tensor)
    predicted_classes_test = (prediction_test >= 0.5).float()
    correct_test = (predicted_classes_test == y_test_tensor).sum()
    accuracy_test = correct_test / y_test_tensor.shape[0]
    print(f"\nTest Accuracy (8:2 split): {accuracy_test.item() * 100:.2f}%")

# ===== 트레인셋 정확도 평가 추가 =====
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)

with torch.no_grad():
    prediction_train = model(x_train_tensor)
    predicted_classes_train = (prediction_train >= 0.5).float()
    correct_train = (predicted_classes_train == y_train_tensor).sum()
    accuracy_train = correct_train / y_train_tensor.shape[0]
    print(f"Train Accuracy (8:2 split): {accuracy_train.item() * 100:.2f}%")
