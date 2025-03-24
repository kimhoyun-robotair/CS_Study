from __future__ import print_function
import torch
from torch.autograd import Variable

x = Variable(torch.randn(1,10))
prev_h = Variable(torch.randn(1,20))
W_h = Variable(torch.randn(20,20))
W_x = Variable(torch.randn(20,10))

q = torch.randn(5,3)
print(q)
print(q.size())

w = torch.Tensor(5,3)
print(w)

e = torch.rand(5,3)
print(q+e)
print(torch.add(q,e))

result = torch.Tensor(5,3)
torch.add(q,e,out=result)
print(result)

e.add_(q)
print(e)

print(q[:,1])

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)


if torch.cuda.is_available():
    q = q.cuda()
    e = e.cuda()
    print(q+e)
