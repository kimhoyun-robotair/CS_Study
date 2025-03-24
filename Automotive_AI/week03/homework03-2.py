import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(777) # for reproducibility

x_train = [[3],[4.5],[5.5],[6.5],[7.5],[8.5],[8],[9],[9.5],[10]]
y_train = [[8.49],[11.93],[16.18],[18.08],[21.45],[24.35],[21.24],[24.84],[25.94],[26.02]]

X = Variable(torch.Tensor(x_train))
Y = Variable(torch.Tensor(y_train))

model = nn.Linear(1, 1, bias=True)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 20 == 0:
        print(step, cost.data.numpy(), model.weight.data.numpy(),
              model.bias.data.numpy())

# 모델 학습 후 1차 함수 출력
slope = model.weight.data.numpy()[0][0]  # 기울기
intercept = model.bias.data.numpy()[0]  # 절편

print(f"학습된 1차 함수: y = {slope:.2f}x + {intercept:.2f}")


# 시각화
plt.scatter(x_train, y_train, label="Train Data", color='blue')  # 실제 데이터 점
plt.plot(x_train, model(X).data.numpy(), label='Linear Regression', color='red')  # 회귀선

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Visualization of Linear Model')

# 그래프 출력
plt.show()
