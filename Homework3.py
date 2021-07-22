# #### 1
import numpy as np
import matplotlib.pyplot as plt

# # 𝑦 =[k*cos(𝑥−𝑎) + 𝑏 ]
# x = np.linspace(-np.pi, np.pi, 201)
#
#
# def y(x, k, a, b):
#     return k * np.cos(x - a) + b
#
# k1, a1, b1 = 3, 2, 2.5
# k2, a2, b2 = 8, 9, 10
#
# plt.plot(x, y(x, k1, a1, b1), color = 'red', label = f'k = {k1}, a = {a1}, b = {b1}')
# plt.plot(x, y(x, k2, a2, b2), color = 'green', label = f'k = {k2}, a = {a2}, b = {b2}')
# plt.legend(frameon = True)
# plt.xlabel('x')
# plt.ylabel('y(x)=k*cos(x-a)+b')
# plt.grid(True)
# plt.show()



##### 2
# #### Напишите код, который будет переводить полярные координаты в декартовы.
# def polar(R, phi):
#     x = R * np.cos(np.radians(phi))
#     y = R * np.sin(np.radians(phi))
#     return round(x, 2), round(y, 2)
# print(polar(10, 30))
#
# #### Напишите код, который будет рисовать график окружности в полярных координатах.
#
# R = 3
# phi = np.linspace(0, 2 * np.pi, 100)
# r = np.linspace(R, R, 100)
# plt.polar(phi, r, color='red')
# plt.show()

####  Напишите код, который будет рисовать график прямой линии в полярных координатах.

# R = 3
# phi = np.linspace(np.radians(160), np.radians(160) , 100)
# phi1 = np.linspace(np.radians(340), np.radians(340) , 100)
# r = np.linspace(0, R, 100)
#
# r1 = np.linspace(0, R, 100)
# plt.polar(phi, r, color='red')
# # plt.show()
# plt.polar(phi1,  r1, color='green')
# plt.polar()
# plt.show()

# R = 3
# x = np.linspace(0, 5)
# # y = x/3
# #plt.ylim(-1, 3)
# plt.plot()
# plt.show()
#
# phi = np.arange(3, 8)
# print(phi)
# R = np.arange(3, 8)
# print(R)
#
# plt.polar(phi, R, color = 'red')
#
# plt.show()

# phi = np.arange(4, 8, 2)
# print(phi)
# R = np.arange(4, 8, 2)
# print(R)
#
# plt.polar(phi, R)
#
# plt.show()



# ######4 - 1
#
# # import numpy as np
# from scipy.optimize import fsolve
# # import matplotlib.pyplot as plt
# # y = x2 – 1
# # y = (exp(x) + x – 1) / x
# x = np.linspace(-2, 3, 200)
# plt.plot(x, x**2 - 1)
# plt.plot(x, (np.exp(x) + x - 1)/x)
# plt.xlabel('x')
# plt.ylabel('y')
# # plt.ylim(-1,5)
# plt.grid(True)
# # plt.axis('scaled')
# plt.show()
#
#
# def equations(p):
#     x, y = p
#     return y - x**2 + 1, np.exp(x) + x*(1-y) - 1
#
# # x1, y1 = fsolve(equations, (2, 2))
#
# x1, y1 = fsolve(equations, (2, 4))
# x2, y2 = fsolve(equations, (-1, 1))
#
# print(x1, y1)
# print(x2, y2)
#
# #
# # x0 = fsolve(f, 1)
# # print(x0)


# ######4 - 2
#
# import numpy as np
from scipy.optimize import fsolve
# import matplotlib.pyplot as plt
# y = x2 – 1
# y = (exp(x) + x – 1) / x
x = np.linspace(-2, 3, 200)
plt.plot(x, x**2 - 1)
plt.plot(x, (np.exp(x) + x - 1)/x)
plt.xlabel('x')
plt.ylabel('y')
# plt.ylim(-1,5)
plt.grid(True)
# plt.axis('scaled')
plt.show()


def equations(p):
    x, y = p
    return y - x**2 + 1, np.exp(x) + x*(1-y) - 1

# x1, y1 = fsolve(equations, (2, 2))

x1, y1 = fsolve(equations, (2, 4))
x2, y2 = fsolve(equations, (-1, 1))

print(x1, y1)
print(x2, y2)

#
# x0 = fsolve(f, 1)
# print(x0)