
import numpy as np

# # #### 1

A = np.array([[1, 2, 3], [4, 0, 6], [7, 8, 9]])
B = np.array([12, 2, 1])
det_A = np.linalg.det(A)

print('Определитель не равен нулю, т.е. мы можем найти решение. А также мы сможем найти обратную матрицу ', round(det_A), 5)

print(np.linalg.solve(A, B))
print('Обратная матрица')
print(np.linalg.inv(A))
print('Проверка обратной матрицы')
print(np.dot(np.linalg.inv(A), A))
print('Решение через обратную матрицу даёт тот же результат')
print(np.dot(np.linalg.inv(A), B))

# # #### 2

A = np.array([[1, 2, -1], [3, -4, 0], [8, -5, 2], [2, 0, -5], [11, 4, -7]])
print('A =', A)
B = np.array([1, 7, 12, 7, 15])
print('\n')
print('B =', B)

# print(det_a = np.linalg.det(A))

X = np.linalg.lstsq(A, B, rcond=None)
print('Псевдорешением является', X[0])
print('Сумма квадтратов невязок каждого уравнения', X[1])
print('Ранг матрицы А =', X[2])
print('\n')
print('Проверка. Подставим наш вектор в систему уравнений')

print('B псевдо =', np.dot(A, X[0]))
print('B =', B)

print('Невязка =', np.dot(A, X[0]) - B)

def Q(x, y, z):
    return ((np.linalg.norm(np.dot(A, [x, y, z]) - B))**2)

print('Проверяем квадрат невязки', round(Q(X[0][0], X[0][1], X[0][2]), 8))

#####3

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[12, 2, 1]])

det_A = np.linalg.det(A)
# print(np.linalg.det(A))
print(f'Определитель равен {det_A}, таким образом система линейных систем с вырожденной матрицей.')


C = np.concatenate((A, B.T), axis=1)
print('Рассмотрим расширенную матрицу: \n', C)

rank_A = np.linalg.matrix_rank(A, 0.0001)
rank_C = np.linalg.matrix_rank(C, 0.0001)


# A = np.array([[1, 2, 3], [4, 5, 6]])
# print(A.shape)

# n = A.shape[1]
# print(n)
print('Найдём количество решений, которое имеет система уравнений')
def K_K(rank_1, rank_2, n):
    n = n

    if rank_1 < rank_2:
        return 'Данная система не имеет решений'
    elif rank_1 == rank_2 and rank_1 == n:
        return 'Данная система имеет единственное решение'
    else:
        return 'Данная система имеет бесконечное количество решений'

print(f'Ранг исходной матрицы = {rank_A}, Ранг расширенной матрицы = {rank_C}. {K_K(rank_A, rank_C, A.shape[1])}')
print('-'*150)
print('Т.к. СЛАУ выше не имеет решений, то изменим вектор B для решения уравнения')
B = np.array([[2, 2, 2]])
C = np.concatenate((A, B.T), axis=1)
print('Расширенная матрица: \n', C)

rank_A = np.linalg.matrix_rank(A, 0.0001)
rank_C = np.linalg.matrix_rank(C, 0.0001)
# print('Количество неизвестных =', n)
print(f'Ранг исходной матрицы = {rank_A}, Ранг расширенной матрицы = {rank_C}. Количество неизвестных = {A.shape[1]}. {K_K(rank_A, rank_C, A.shape[1])}')

# print(np.linalg.solve(A, B.flatten()))

# X = np.linalg.lstsq(A, B.flatten(), rcond=None)[0]
# print(X)



# print(np.linalg.solve(A, B))

X = np.linalg.lstsq(A, B.flatten(), rcond=None)
print('Решением является', X[0])
print('Проверка. Подставим наш вектор в систему уравнений')
#
# print('B псевдо =', np.dot(A, X[0]))
# print('B =', B)
#
print('Невязка =', np.dot(A, X[0]) - B)

# A = np.array([[1, 2, 3], [4, 5, 6]])
# B = np.array([[12, 2, 1]])
# Y = np.linalg.solve(A, B)
# print(Y)


import scipy
import scipy.linalg
####4


A = np.array([[1, 2, 3], [2, 16, 21], [4, 28, 73]])
print(A)

P, L, U = scipy.linalg.lu(A)
print(f'P: \n{P}')
print(f'L: \n{L}')
print(f'U: \n{U}')
print('-' * 150)
print(f'P * A - L * U: \n{np.dot(P, A) - np.dot(L, U)}')
print('Требуется, чтобы P * A = L * U. У нас это не так')
# print(f'P*A:\n{P * A}')
print(np.dot(P, A))
# # print(f'L*U:\n{L * U}')
print(np.dot(L, U))
#
#
det_A = np.linalg.det(A)
print('Определитель', round(det_A, 2))

print('-' * 150)
print('Подставим рандомный вектор В')
B = np.array([2, 2, 2])
X = np.linalg.solve(A, B)
print(X)
print('Невязка =', np.dot(A, X) - B)

####5

import matplotlib.pyplot as plt

A = np.array([[1, 2, -1], [8, -5, 2]])
B = np.array([1, 12])

x = np.linspace(0.5, 2.4, 200)
y = 10 * x - 14
z = x + 2 * y - 1


def Q(x, y, z):
    return (x**2 + y**2 + z**2)


plt.plot(x, Q(x, y, z))
plt.xlabel('x')
plt.ylabel('Q(x, y, z)')
plt.grid(True)
plt.show()

X = np.linalg.lstsq(A, B, rcond=None)

print('Моё решение = ')
x = 1.37
y = 10 * x - 14
z = x + 2 * y - 1

X_My = np.array([x, y, z])
print(X_My)


print('Точное решение', X[0])
print(np.dot(A, X[0]), f'Это близко к значению B {B}')
# A = np.array([[1, 2, -1], [8, -5, 2]])
# B = np.array([1, 12])
# print()


######6
import matplotlib.pyplot as plt

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[2, 5, 11]])

print('Исходная матрица A\n', A)

det_A = np.linalg.det(A)
print(f'Определитель равен {det_A}, таким образом система линейных систем с вырожденной матрицей. Система имеет'
      f'бесконечное множество псевдорешений минимизирующих норму невязки')

# C = np.concatenate((A, B.T), axis=1)
# print('Рассмотрим расширенную матрицу: \n', C)

print('Разложим при помощи QR разложения.')
Q, R = np.linalg.qr(A)
print(f'Q: \n{Q}')
print(f'R: \n{R}')

# def K_K(rank_1, rank_2, n):
#     n = n
#     if rank_1 < rank_2:
#         return 'Данная система не имеет решений'
#     elif rank_1 == rank_2 and rank_1 == n:
#         return 'Данная система имеет единственное решение'
#     else:
#         return 'Данная система имеет бесконечное количество решений'
#
# rank_A = np.linalg.matrix_rank(A)
# rank_C = np.linalg.matrix_rank(C)
#
# print(K_K(rank_A, rank_C, A.shape[1]))
#
# X = np.linalg.lstsq(A, B.flatten(), rcond=None)[0]
# print(X)

print(f'Q*R = \n{np.dot(Q, R)}\nПолучаем матрицу близкую к исходной. Соответственно разложение верно')
print('Проверим, что матрица Q ортогональная. Умножим транспонированную матирицу на саму себя. Должны получить единичную.')
print(f'Qt * Q \n{np.dot(np.transpose(Q), Q)}')
# print('\nR1*X1 = Qt*B - R2*X2')
# print('\nВыделим матрицы R1(верхнетреугольная матрица)  и B1')

R1 = R[:2, :2]
print(f'R1: \n{R1}')

# print(B)
B1 = np.dot(np.transpose(Q), B.flatten())[:2]
print(f'B1: \n{B1}')

X1 = np.linalg.solve(R1, B1)
print(f'X1:\n{X1}')

X = np.append(X1, 0)
print(f'Псевдорешение для исходной системы X:\n{X}')
print(f'Норма:\n{np.linalg.norm(X)}')

print('Невязка = ', np.linalg.norm(np.dot(A, X) - B))

print('Найдём нормальное псевдорешение')

# x = np.linspace(0, 2, 200)
# y = 3 - 2 * x
# z = (2 - x - 2*y)/3
#
# def Q(x, y, z):
#     return (x**2 + y**2 + z**2)
#
#
# plt.plot(x, Q(x, y, z))
# plt.xlabel('x')
# plt.ylabel('Q(x, y, z)')
# plt.grid(True)
# plt.show()
#
# x = 1.19
# y = 3 - 2 * x
# z = (2 - x - 2*y)/3
#
# X_My1 = np.array([x, y, z])
# print('Моё решение X_My1', X_My1)
# print('Проверим решение, подставив в систему уравнений', np.dot(A, X_My1))
#
# X = X_My1
# print('Норма вектора X_My',np.linalg.norm(X))
# print('Норма вектора невязки исходной системы X_My', np.linalg.norm(np.dot(A, X) - B))
#
# print('-' * 150)
# x = 0.99
# y = 2.5 - 2 * x
# z = x - 1
#
#
# X_My2 = np.array([x, y, z])
# print('Моё решение X_My2', X_My2)
# print('Проверим решение, подставив в систему уравнений', np.dot(A, X_My2))
#
# X = X_My2
# print('Норма вектора X_My',np.linalg.norm(X))
# print('Норма вектора невязки исходной системы X_My', np.linalg.norm(np.dot(A, X) - B))
#
# print('-' * 150)

X_res = np.linalg.lstsq(A, B.flatten(), rcond=None)
print(X_res[0])
print('Проверим решение, подставив в систему уравнений', np.dot(A, X_res[0]))


X = X_res[0]
print('Норма вектора X:', np.linalg.norm(X))
print('Норма вектора невязки исходной системы', np.linalg.norm(np.dot(A, X) - B))