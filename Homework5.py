# # #### 1
import numpy as np
import matplotlib.pyplot as plt

red = [i for i in range(0, 36) if(i%2 == 0)]
black = [i for i in range(36) if(i%2 != 0)]

# print(red)
# print(black)
# for i in range(0, 5):
#
#     a = input()
#     x = np.random.uniform(0, 37)
#     # print(int(x))
#     if int(x) == 0:
#         print("zero")
#     # elif (red.index(int(x)) == True):
#     #     print("red")
#     elif int(x) in red:
#         print(x, 'red')
#     else:
#         print(x, 'black')

r = 0
b = 0
for i in range(0, 5):

    a = input('Enter enter')
    x = np.random.uniform(0, 37)

    if int(x) == 0:
        print("zero")
    elif int(x) % 2 == 0:
        print(int(x), 'red')
        r += 1
    else:
        print(int(x), 'black')
        b += 1

print(r, b)

### 2 Напишите код, проверяющий любую из теорем сложения или умножения вероятности на примере рулетки или подбрасывания монетки.


import numpy as np
r = 0
b = 0
zero = 0

number_of_throws = 100 #int(input('Enter number of throws: '))
for i in range(0, number_of_throws):
    x = np.random.uniform(0, 37)

    if int(x) == 0:
        # print("zero")
        zero += 1
    elif int(x) % 2 == 0:
        # print(int(x), 'red')
        r += 1
    else:
        # print(int(x), 'black')
        b += 1

# def proverka(r, b, z):
#     if (r/number_of_throws + b/ number_of_throws + z/number_of_throws) <1:
#         return True
#     else:
#         return False

print(f'Частота вероятоности выпадения красного = {r/number_of_throws}, частота вероятности выпадения чёрного = {b/number_of_throws}. Частота вероятности выпадения зеро = {zero/number_of_throws} '
      f'Это число должно быть равно 1. {(r/number_of_throws + b/number_of_throws + zero/number_of_throws) == 1}')



### 2-2
all = []
# for n in range(10):
#     print(np.random.rand(10))

# print([np.random.randint(0, 100, 10) for i in range(100)])

# print(np.random.randint(1, 100, 24))
for n in range(10):
    a = np.random.randint(1, 100, 24)
    all.append(a)
    print(a)
    # all = np.array(a)
# print(all)
num_bins = 5
n, bins, patches = plt.hist([np.sum(x) for x in all], num_bins)
plt.xlabel('summa')
plt.ylabel('Probability')
plt.title('Histogram')
# plt.hist([np.sum(x) for x in all])
plt.show()

# k,m = 0, 0
# n = 100
# for i in range(0, n):
#     x = np.random.uniform(0, 10)
#     if x < 5:
# #        print("орел")
#         k = k + 1
#     else:
# #        print("решка")
#         m = m + 1
# print(f'Количество орлов = {k}, количество решек = {m}')



# n = 10
# m = 10
# total = np.array([np.random.randint(0, 100, m) for _ in range(n)])
# print(total)
# # построим гистограмму распредления суммы случайных величин:
# plt.hist(np.array([np.sum(x) for x in total]))
# plt.show()


### 2-2 another

# import random
# summa = 0
# x_list = []
# for n in range(10):
#     x = np.random.randint(0, 300)
#     x_list.append(x)
#     print('x = ', x)
#     summa += summa
#
#
# print(np.random.choice(x_list))
# plt.hist(summa)
# plt.show()


### 3


import numpy as np
import itertools

k, n = 0, 10000
a = np.random.randint(0, 2, n)
b = np.random.randint(0, 2, n)
c = np.random.randint(0, 2, n)
d = np.random.randint(0, 2, n)
x = a + b + c + d
for i in range(0, n):
    if x[i] == 2:
        k = k + 1
#print(a, b, c, d)
#print(x)
print(k, n, k/n)


def Monte_Carlo(k, n):
    a = np.random.randint(0, 2, n)
    b = np.random.randint(0, 2, n)
    c = np.random.randint(0, 2, n)
    d = np.random.randint(0, 2, n)
    x = a + b + c + d
    for i in range(0, n):
        if x[i] == 2:
            k = k + 1
    return k / n

# print(Monte_Carlo(0, 10000))

# при четырёх испытаниях выпадет 2 успеха
def Bernulli(k, n, p = 0.5):
    q = 1 - p
    C = np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n - k))
    return C * np.power(p, k) * np.power(q, n - k)

# print(Bernulli(2, 4))

print(f'Monte_Carlo = {np.round(Monte_Carlo(0, 1000), 3)}, Bernulli = {np.round(Bernulli(2, 4), 3)}')


#
##### 3 - 2

def Monte_Carlo(k, n):
    a = np.random.randint(0, 2, n)
    b = np.random.randint(0, 2, n)
    c = np.random.randint(0, 2, n)
    d = np.random.randint(0, 2, n)
    f = np.random.randint(0, 2, n)
    g = np.random.randint(0, 2, n)

    x = a + b + c + d + f + g
    for i in range(0, n):
        if x[i] == 3:
            k = k + 1
    return k / n


# при четырёх испытаниях выпадет 2 успеха вероятность выпадения двух успехов на 4-ех испытаниях
def Bernulli(k, n, p = 0.5):
    q = 1 - p
    C = np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n - k))
    return C * np.power(p, k) * np.power(q, n - k)

print(f'Monte_Carlo = {np.round(Monte_Carlo(0, 1000), 3)}, Bernulli = {np.round(Bernulli(3, 6), 3)}')


####4


import itertools


# for p in itertools.product("01", repeat = 2):
#     print(''.join(p))

# for p in itertools.product("0123",repeat=3):
#     print(''.join(p))

# для других k и n
# размещения
# for p in itertools.permutations("01234", 3):
#     print(''.join(str(x) for x in p))
# комбинации - сочетания
# for p in itertools.combinations("01234", 2):
#      print(''.join(p))

# for p in itertools.product("01234",repeat=4):
#     print(''.join(p))


#####5
# коэффициент корреляции
def R(x, y):
    a = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    b = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y), 2)))
    return a / b

n = 100
r = 0.7
x = np.random.rand(n)
y = r*x + (1 - r)*np.random.rand(n)
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

a = (np.sum(x)*np.sum(y) - n*np.sum(x*y))/(np.sum(x)*np.sum(x) - n*np.sum(x*x))
b = (np.sum(y) - a*np.sum(x))/n

A = np.vstack([x, np.ones(len(x))]).T
a1, b1 = np.linalg.lstsq(A, y, rcond = -1)[0]
print(a, b)
print(a1, b1)
plt.plot([0, 1], [b, a + b])
plt.show()
print(f'R: {np.round(R(x, y), 4)}')