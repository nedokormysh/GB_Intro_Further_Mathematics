import numpy as np
###5.2
a = np.array([[1, 2, 3], [4, 0, 6], [7, 8, 9]])

print(a)
print(np.linalg.det(a))

# 5.3

import numpy as np

a = np.array([[1, 5, 3, 4], [2, 10, 6, 8], [1/2, 5/2, 3/2, 2], [3, 15, 9, 12]])
print(a)
print(np.linalg.matrix_rank(a))

#5.4

a = np.array([1, 5])
b = np.array((2, 8))

print(np.dot(a, b))

import matplotlib.pyplot as plt



U, V = a[0], a[1]
X, Y = np.array([1, 0]), np.array([0, 0])
U, V = np.array([0, 5]), np.array([2, 8])
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)

plt.xlim(-2, 10)
plt.ylim(-2, 10)
plt.grid(True)
plt.show()