import numpy as np

def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a-C)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
