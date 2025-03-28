import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

xy = np.loadtxt("/home/kimhoyun/CS_Study/Automotive_AI/week04/examdata.txt", delimiter=",", dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x_data = Variable(torch.from_numpy(x_data))
y_data = Variable(torch.from_numpy(y_data))

model = nn.Linear(3, 1, bias=True)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for step in range(2001):
    optimizer.zero_grad()

    hypothesis = model(x_data)
    cost = criterion(hypothesis, y_data)
    cost.backward()
    optimizer.step()


    if step % 200 == 0:
        print(step, "Cost: ", cost.data.numpy(), "\nPrediction: \n",
              hypothesis.data.numpy())

print("Your score will be ", model(Variable(torch.Tensor([[100,70,101]]))).data.numpy())
print("Other scores will be ", model(Variable(torch.Tensor([[60,70,110], [90,100,80]]))).data.numpy())
