import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)

xy = np.loadtxt("/home/kimhoyun/CS_Study/Automotive_AI/week04/lecture6_data.txt", dtype=np.float32)
x_data = xy[:, 2:-1]
y_data = xy[:, [-1]]

x_data = Variable(torch.from_numpy(x_data))
y_data = Variable(torch.from_numpy(y_data))

model = nn.Linear(2, 1, bias=True)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(x_data)
    cost = criterion(hypothesis, y_data)
    cost.backward()
    optimizer.step()


    if step % 10 == 0:
        print(step, "Cost: ", cost.data.numpy(), "\nPrediction: \n",
              hypothesis.data.numpy())

print("Your Final exam score will be ", model(Variable(torch.Tensor([[57.8, 21]]))).data.numpy())
print("Other Scores Will be ", model(Variable(torch.Tensor([[23.0, 43.3], [42.5, 59.4]]))).data.numpy())
