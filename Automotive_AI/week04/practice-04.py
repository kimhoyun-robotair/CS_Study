import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

x_data = np.array([[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]], dtype=np.float32)
y_data = np.array([[0],[0],[0],[1],[1],[1]], dtype=np.float32)

X = Variable(torch.from_numpy(x_data))
Y = Variable(torch.from_numpy(y_data))

linear = nn.Linear(2,1, bias=True)
sigmoid = nn.Sigmoid()

model = nn.Sequential(linear, sigmoid)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = -(Y*torch.log(hypothesis) + (1-Y)*torch.log(1-hypothesis)).mean()

    cost.backward()
    optimizer.step()

    if step % 200 == 0:
        print(step, cost.data.numpy())

predicted = (model(X).data > 0.5).float()
accuracy = (predicted == Y.data).float().mean()

# 학습된 가중치와 바이어스 출력
W = linear.weight.data
b = linear.bias.data
print(f"Weight (W): {W.numpy()}")
print(f"Bias (b): {b.numpy()}")

# 새로운 입력 데이터 정의
new_data = np.array([[4,2]], dtype = np.float32)
new_X = torch.from_numpy(new_data)  # 넘파이 → 파이토치 텐서로 변환

# 학습된 모델로 새로운 데이터 예측
with torch.no_grad():  # 그래디언트 추적 없이 연산 (추론 시 사용)
    new_pred = model(new_X)  # 예측 확률
    predicted_class = (new_pred > 0.5).float()  # 분류 결과 (0 또는 1)

# 새로운 입력에 대한 예측 결과 출력
print(f"입력: {new_data}")
print(f"예측된 확률: {new_pred.item():.4f}")
