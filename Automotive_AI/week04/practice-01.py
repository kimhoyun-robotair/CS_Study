import torch
from torch.autograd import Variable
import torch.nn as nn

torch.manual_seed(777)

x_data = [[74., 80., 75.], [93., 88., 93.], [89., 91., 90.],
          [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

X = Variable(torch.Tensor(x_data))
Y = Variable(torch.Tensor(y_data))

model = nn.Linear(3, 1, bias=True)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, "Cost: ", cost.data.numpy(), "\nPrediction: \n",
              hypothesis.data.numpy())
