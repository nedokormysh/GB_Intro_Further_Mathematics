# #### 1
# import math
# ### решение для векторов данных в условии
# x = [10, 10, 10]
# y = [0, 0, -10]
# result_vector = []
#
# for i in range (0, len(x)):
#     result_vector.append(x[i] + y[i])
# print(result_vector)
#
#
# def length(li):
#     length_temp = 0
#     for l in li:
#         length_temp += l ** 2
#     return math.sqrt(length_temp)
#
# print('Длина вектора X =', round(length(x), 2))
# print('Длина вектора Y =', round(length(y), 2))
# print('Длина результирующего вектора =', round(length(result_vector), 2))

# # реализация нахождения длины для произвольного вектора
# vector = []
# while True:
#     s = input('Enter numbers. To exit enter Q.: \n')
#     if s.title() == 'Q':
#         break
#     else:
#         try:
#             vector.append(float(s))
#         except ValueError:
#             print('Enter numbers or q for quit')
#
#         sqrnumbers = []
#         for el in vector:
#             sqrnum = el**2
#             sqrnumbers.append(round(sqrnum, 2))
# summa = sum(sqrnumbers)
#
# print(f'Your vector is {vector}')
# print(sqrnumbers)
# print('Length of vector is', round(math.sqrt(summa), 2))

# #### 2
#
# ### изменение масштабной сетки для правильного отображения перпенидикулярности прямых для задания номер 2
# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.linspace(-5, 5, 21)
# y = 3 * x + 1
# y2 = (-1/3) * x + 1
#
# plt.plot(x, y)
# plt.plot(x, y2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('equal')
# plt.show()

####3

import numpy as np
import matplotlib.pyplot as plt



# #### круг
#
# # x = np.linspace(0, 10, 21)
# x1 = []
# y1 = []
# x2 = []
# y2 = []
#
# R = 100
# for i in range(-100, 101):
#     x_ = i
#     x1.append(x_)
#     # x2.append(i)
#     y1.append(np.sqrt(R**2 - (x_**2)))
#     # y2.append(-y1)
#     y2.append(-np.sqrt(R**2 - (x_**2)))
#     # y2 =
#
# plt.figure(figsize = (6, 6))
# plt.plot(x1, y1, color = 'r')
# plt.plot(x1, y2, color = 'g')
# plt.plot()
# # plt.plot(x2, y1, color = 'g')
# plt.axis('scaled')
# plt.plot()
# plt.show()

# x = []
# x2 = []
# y = []
# y2 = []
#
# for i in range(1500):
#     r = 1000
#     x.append(-i)
#     x2.append(i)
#     y.append(np.sqrt(r ** 2 - i ** 2))
#     y2.append(-np.sqrt(r ** 2 - i ** 2))
#
# plt.figure(figsize=(6, 6))
# plt.plot(x, y, color='r')
# plt.plot(x, y2, color='r')
# plt.plot(x2, y2, color='r')
# plt.plot(x2, y, color='r')
# plt.axis('scaled')
# plt.show()

# #### эллипс

## k = 5
##y = k/x
##
## plt.plot(x, y)
## plt.show()

# x = []
# y1 = []
# y2 = []
#
# a = 40
# b = 20
# for i in range(-40, 41):
#     x_ = i
#     x.append(x_)
#     # y = (b**2(a**2 - x_**2))/a**2
#     y = b**2 - ((b**2 * x_**2)/a**2)
#     y1.append(np.sqrt(y))
#     y2.append(-np.sqrt(y))
# plt.plot(x, y1, color = 'r')
# plt.plot(x, y2, color = 'g')
# plt.axis('scaled')
# plt.show()

# #### гипербола

# x = []
# y1 = []
# y2 = []
#
# a = 60
# b = 30
# for i in range(-60, 61):
#     x_ = i
#     x.append(x_)
#     # y = (b**2(a**2 - x_**2))/a**2
#     y = b**2 + ((b**2 * x_**2)/a**2)
#     y1.append(np.sqrt(y))
#     y2.append(-np.sqrt(y))
# plt.plot(x, y1, color = 'r')
# plt.plot(x, y2, color = 'g')
# plt.axis('scaled')
# plt.show()

# ####### 5 - 1
# from pylab import *
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = figure()
#
# # ax = Axes3D(fig)
# # ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(1, 1, 1, projection = '3d')
# # fig.add_axes(ax)
# X = np.arange(-20, 20, 2)
# Y = np.arange(-20, 20, 2)
# X, Y = np.meshgrid(X, Y)
# Z = 2 * X + 5 * Y
#
# ax.plot_wireframe(X, Y, Z, label = '1: Ax + By + Cz = 0', colors = 'blue')
# ax.plot_wireframe(X, Y, Z + 200, label = '2: Ax + By + Cz + D = 0', colors = 'red')
# # ax.plot_surface(X, Y, Z, label = '1',  color='blue')
# # ax.plot_surface(X, Y, Z + 50, label = '2', color='red')
# ax.scatter(0, 0, 0,'z',50,'green', label='Координата: 0, 0, 0')
# plt.legend(frameon = True)
# show()


# ####### 5 - 2
#
# #### эллиптический параболоид
#
# from pylab import *
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = figure()
# ax = fig.add_subplot(1, 1, 1, projection = '3d')
#
#
# X = np.arange(-20, 20, 2)
# Y = np.arange(-20, 20, 2)
# X, Y = np.meshgrid(X, Y)
#
#
# a = 15
# b = 30
#
# Z = X**2/a**2 + Y**2/b**2
# ax.plot_wireframe(X, Y, Z, label = '1: Ax + By + Cz = 0', colors = 'blue')
# show()


# #### гиперболический параболоид
# #
# from pylab import *
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = figure()
# ax = fig.add_subplot(1, 1, 1, projection = '3d')
#
#
# X = np.arange(-20, 20, 2)
# # print(X)
# Y = np.arange(-20, 20, 2)
# X, Y = np.meshgrid(X, Y)
# # print(Y)
#
#
# a = 15
# b = 30
#
# Z = X**2/a**2 - Y**2/b**2
# ax.plot_wireframe(X, Y, Z, label = '1: Ax + By + Cz = 0', colors = 'blue')
# ax.plot_surface()
# show()


### zapolnenie

from pylab import *
from mpl_toolkits.mplot3d import Axes3D

fig = figure()
ax = fig.add_subplot(1, 1, 1, projection = '3d')

X = np.arange(-20, 20, 2)
# print(X)
Y = np.arange(-20, 20, 2)
X, Y = np.meshgrid(X, Y)
# print(Y)


a = 15
b = 30

Z = X**2/a**2 - Y**2/b**2
# ax.plot_wireframe(X, Y, Z, label = '1: Ax + By + Cz = 0', colors = 'blue')
ax.plot_surface(X, Y, Z, color = 'yellow')
show()

### lines

from pylab import *
from mpl_toolkits.mplot3d import Axes3D

fig = figure()
ax = fig.add_subplot(1, 1, 1, projection = '3d')


X = np.arange(-20, 20, 2)
# print(X)
Y = np.arange(-20, 20, 2)
# X, Y = np.meshgrid(X, Y)
# print(Y)


a = 15
b = 30

Z = X**2/a**2 - Y**2/b**2
# ax.plot_wireframe(X, Y, Z, label = '1: Ax + By + Cz = 0', colors = 'blue')
ax.plot(X, Y, Z, color = 'yellow')
show()