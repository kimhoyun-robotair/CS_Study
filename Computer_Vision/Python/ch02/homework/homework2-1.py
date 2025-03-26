"""
numpy ndarray 안에 다양한 자료형을 넣고, 어떻게 작동하는지 확인해보자.
"""

import numpy as np

a = np.array([400, 52, 'tiger', '24', 230])
print(a)
print(type(a))
print(a.shape)
a.sort()
print(a)
print(a.dtype)

"""
유니코드 문자열로 통일되서 나온다.
a.dtpye을 입력하면 <U21인데, 이는 유니코드 문자열을 의미한다.
"""
