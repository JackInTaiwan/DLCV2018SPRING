import numpy as np



x = np.array([[1,2,3], [4,5,6], [7,8,9]])
y = np.array([[0], [0], [1]])
print (np.matmul(x, y))

a = np.array([1,2,3,4]).reshape(-1, 1)
b = np.array([0,1,2,2]).reshape(-1, 1)
print (np.sum((a-b) ** 2))