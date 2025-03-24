import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777) # for reproducibility

x_train = [[10],[9],[3],[2]]
y_train = [[90],[80],[50],[30]]

X = Variable(torch.Tensor(x_train))
Y = Variable(torch.Tensor(y_train))

model = nn.Linear(1, 1, bias=True)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(200001):
    optimizer.zero_grad()
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()
"""
    if step % 20 == 0:
        print(step, cost.data.numpy(), model.weight.data.numpy(),
              model.bias.data.numpy())"""

predicted = model(Variable(torch.Tensor([[3.5], [4.5]])))
print(predicted.data.numpy())
