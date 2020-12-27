import sympy as sp
from sympy import Eq, Function, Matrix, factor, dsolve, Integral, integrate
from sympy.solvers.ode.systems import dsolve_system
from sympy.interactive import printing
from sympy import simplify, sympify
from numpy.linalg import matrix_rank
from numpy import *
import numpy as np
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sympy.plotting import plot
printing.init_printing(use_latex='mathjax')
sp.init_printing()
import time

def solve_linear_eq(ex1, ex2, ex3):
    return sp.solve([ex1, ex2, ex3], (C1, C2, C3))

def keep_only_eq(ex):
    return sympify(str(ex)[3:-4])

def change_var(ex, result):
    ex1 = factor(ex.subs([(C1, result[C1])]))
    ex2 = factor(ex1.subs([(C2, result[C2])]))
    ex3 = factor(ex2.subs([(C3, result[C3])]))
    return ex3

def change_var_not_control(ex, result):
    ex1 = factor(ex.subs([(C1, result[C1])]))
    ex2 = factor(ex1.subs([(C2, result[C2])]))
    return ex2

def make_z(ex1, ex2, ex3):
    z1 = []

    z1.append(ex1)
    z1.append(ex2)
    z1.append(ex3)

    return z1

def make_z_not_control(ex1, ex2):
    z1 = []
    z1.append(ex1)
    z1.append(ex2)
    return z1

def multiple_replace(target_str, replace_values):
    # получаем заменяемое: подставляемое из словаря в цикле
    for i, j in replace_values.items():
        # меняем все target_str на подставляемое
        target_str = target_str.replace(i, j)
    return target_str

result = []

start_time = time.time()
#розв'язуємо систему диф. рівнянь
x1, x2, x3 = sp.symbols('x1 x2 x3', cls=Function)
t = sp.Symbol('t')

a = float(input("Введіть параметр a: "))
b = float(input("Введіть параметр b: "))
g = float(input("Введіть параметр g: "))

eq1 = Eq(x1(t).diff(t), -a*x1(t) - b*x2(t))
eq2 = Eq(x2(t).diff(t), -b*x1(t) + g*x2(t))
eq3 = Eq(x3(t).diff(t), a*x1(t) + g*x2(t))

#перевіримо на повну керованість за другим  критерієм
A = np.matrix([[-a, -b, 0.], [-b, g, 0.], [a, g, 0.]])

B = np.matrix([[1., 0.], [0., 1.], [0., 0.]])

AAB = np.concatenate((B, A*B), axis=1)

rank = matrix_rank(AAB)

if rank == 3:
    print("Система є цілком керованою")
    solved_diff_system = dsolve_system((eq1, eq2, eq3))
    for i in range(len(solved_diff_system)):
        for j in range(len(solved_diff_system[i])):
            result.append(str(solved_diff_system[i][j])[10:-1])  # друк чисто відповіді

    C1, C2, C3 = sp.symbols('C1, C2, C3')

    # ініціалізуємо три системи лін. рівнянь
    # перший індекс показує який номер системи
    # друший індекс показує який номер рівняння
    lin_eq_1_1 = sp.Eq(sympify(result[0]), 1)
    lin_eq_1_2 = sp.Eq(sympify(result[1]), 0)
    lin_eq_1_3 = sp.Eq(sympify(result[2]), 0)

    lin_eq_2_1 = sp.Eq(sympify(result[0]), 0)
    lin_eq_2_2 = sp.Eq(sympify(result[1]), 1)
    lin_eq_2_3 = sp.Eq(sympify(result[2]), 0)

    lin_eq_3_1 = sp.Eq(sympify(result[0]), 0)
    lin_eq_3_2 = sp.Eq(sympify(result[1]), 0)
    lin_eq_3_3 = sp.Eq(sympify(result[2]), 1)

    # розв'зуємо систему лін. рівнянь відносно C1, C2, C3
    result_1 = solve_linear_eq(lin_eq_1_1, lin_eq_1_2, lin_eq_1_3)
    result_2 = solve_linear_eq(lin_eq_2_1, lin_eq_2_2, lin_eq_2_3)
    result_3 = solve_linear_eq(lin_eq_3_1, lin_eq_3_2, lin_eq_3_3)

    # необхідно внаслідок форматування при переході від str() до sympy.core.expression
    ex_1_1 = keep_only_eq(lin_eq_1_1)
    ex_1_2 = keep_only_eq(lin_eq_1_2)
    ex_1_3 = keep_only_eq(lin_eq_1_3)
    ex_2_1 = keep_only_eq(lin_eq_2_1)
    ex_2_2 = keep_only_eq(lin_eq_2_2)
    ex_2_3 = keep_only_eq(lin_eq_2_3)
    ex_3_1 = keep_only_eq(lin_eq_3_1)
    ex_3_2 = keep_only_eq(lin_eq_3_2)
    ex_3_3 = keep_only_eq(lin_eq_3_3)

    # створюємо вектори z1, z2, z3 фундаментальної системи
    vect_z1 = make_z(change_var(ex_1_1, result_1), change_var(ex_1_2, result_1), change_var(ex_1_3, result_1))
    vect_z2 = make_z(change_var(ex_2_1, result_2), change_var(ex_2_2, result_2), change_var(ex_2_3, result_2))
    vect_z3 = make_z(change_var(ex_3_1, result_3), change_var(ex_3_2, result_3), change_var(ex_3_3, result_3))

    # фундаментальна матриця
    Q = Matrix([vect_z1, vect_z2, vect_z3])
    print(Q)

    #розв'язуємо систему диф. рівнянь
    k1, k2, k3, k4, k5, k6 = sp.symbols('k1 k2 k3 k4 k5 k6')

    matrix_k = Matrix([[k1, k2, k3], [k2, k4, k5], [k3, k5, k6]])
    tmp = A*matrix_k + matrix_k*A.T + B*B.T

    #замінюємо символьні значення k_i на функції z_i(t)
    matrix_for_diff = []
    z1, z2, z3, z4, z5, z6 = sp.symbols('z1 z2 z3 z4 z5 z6', cls=Function)
    replace_values = {'k1': 'z1(t)', 'k2': 'z2(t)', 'k3': 'z3(t)', 'k4': 'z4(t)', 'k5': 'z5(t)', 'k6': 'z6(t)'}

    for i in tmp:
        matrix_for_diff.append(sympify(multiple_replace(str(i), replace_values)))

    for i in matrix_for_diff:
        print(i)

    eq_z_1 = Eq(matrix_for_diff[0], z1(t).diff(t))
    eq_z_2 = Eq(matrix_for_diff[1], z2(t).diff(t))
    eq_z_3 = Eq(matrix_for_diff[2], z3(t).diff(t))
    eq_z_4 = Eq(matrix_for_diff[3], z2(t).diff(t))
    eq_z_5 = Eq(matrix_for_diff[4], z4(t).diff(t))
    eq_z_6 = Eq(matrix_for_diff[5], z5(t).diff(t))
    eq_z_7 = Eq(matrix_for_diff[6], z3(t).diff(t))
    eq_z_8 = Eq(matrix_for_diff[7], z5(t).diff(t))
    eq_z_9 = Eq(matrix_for_diff[8], z6(t).diff(t))

    #розв'язуємо систему диф. рівнянь
    print()
    solved_diff_system_z = []
    C1, C2, C3 = sp.symbols('C1, C2, C3')
    system = [eq_z_1, eq_z_4, eq_z_5, eq_z_6, eq_z_7, eq_z_9]
    ics = {z1(0): 0, z2(0): 0, z3(0): 0, z4(0): 0, z5(0): 0, z6(0): 0}
    tmp_system_z = dsolve(system, [z1(t),  z2(t), z3(t), z4(t), z5(t), z6(t)], ics=ics)

    for j in tmp_system_z:
        solved_diff_system_z.append(simplify(str(j)[10:-1]))

    for i in solved_diff_system_z:
        print(i)

    print()

    PHI = Matrix([[solved_diff_system_z[0], solved_diff_system_z[1], solved_diff_system_z[2]],
                 [solved_diff_system_z[1], solved_diff_system_z[2], solved_diff_system_z[3]],
                 [solved_diff_system_z[3], solved_diff_system_z[4], solved_diff_system_z[5]]])

    # кінцева точка
    x_T = Matrix([20, 20, 20])

    # початкова точка
    x_0 = Matrix([0, 0, 0])
else:
    print("Система не є цілком керованою")
    solved_diff_system = dsolve_system((eq1, eq2))
    for i in range(len(solved_diff_system)):
        for j in range(len(solved_diff_system[i])):
            result.append(str(solved_diff_system[i][j])[10:-1])  # друк чисто відповіді

    print(solved_diff_system)

    C1, C2 = sp.symbols('C1, C2')

    # ініціалізуємо дві системи лін. рівнянь
    # перший індекс показує який номер системи
    # друший індекс показує який номер рівняння
    lin_eq_1_1 = sp.Eq(sympify(result[0]), 1)
    lin_eq_1_2 = sp.Eq(sympify(result[1]), 0)

    lin_eq_2_1 = sp.Eq(sympify(result[0]), 0)
    lin_eq_2_2 = sp.Eq(sympify(result[1]), 1)
    
    # розв'зуємо систему лін. рівнянь відносно C1, C2
    result_1 = sp.solve([lin_eq_1_1, lin_eq_1_2], (C1, C2))
    result_2 = sp.solve([lin_eq_2_1, lin_eq_2_2], (C1, C2))

    # необхідно внаслідок форматування при переході від str() до sympy.core.expression
    ex_1_1 = keep_only_eq(lin_eq_1_1)
    ex_1_2 = keep_only_eq(lin_eq_1_2)
    ex_2_1 = keep_only_eq(lin_eq_2_1)
    ex_2_2 = keep_only_eq(lin_eq_2_2)

    # створюємо вектори z1, z2 фундаментальної системи
    vect_z1 = make_z_not_control(change_var_not_control(ex_1_1, result_1), change_var_not_control(ex_1_2, result_1))
    vect_z2 = make_z_not_control(change_var_not_control(ex_2_1, result_2), change_var_not_control(ex_2_2, result_2))

    # фундаментальна матриця
    Q = Matrix([vect_z1, vect_z2])
    print("Фундаментальна матриця: ")
    print(Q)

    # розв'язуємо систему диф. рівнянь
    k1, k2, k3 = sp.symbols('k1 k2 k3')

    matrix_k = Matrix([[k1, k2], [k2, k3]])

    A = np.matrix([[-a, -b], [-b, g]])
    B = np.matrix([[1., 0.], [0., 1.]])

    tmp = A * matrix_k + matrix_k * A.T + B * B.T

    # замінюємо символьні значення k_i на функції z_i(t)
    matrix_for_diff = []
    z1, z2, z3 = sp.symbols('z1 z2 z3', cls=Function)
    replace_values = {'k1': 'z1(t)', 'k2': 'z2(t)', 'k3': 'z3(t)'}

    for i in tmp:
        matrix_for_diff.append(sympify(multiple_replace(str(i), replace_values)))

    for i in matrix_for_diff:
        print(i)

    eq_z_1 = Eq(matrix_for_diff[0], z1(t).diff(t))
    eq_z_2 = Eq(matrix_for_diff[1], z2(t).diff(t))
    eq_z_3 = Eq(matrix_for_diff[3], z3(t).diff(t))

    # розв'язуємо систему диф. рівнянь
    print()
    solved_diff_system_z = []
    C1, C2, C3 = sp.symbols('C1, C2, C3')
    system = [eq_z_1, eq_z_2, eq_z_3]
    tmp_system_z = dsolve(system, [z1(t), z2(t), z3(t)])

    print(tmp_system_z)
    tmp_system_z[0] = tmp_system_z[0].subs([(t, 0)])
    tmp_system_z[1] = tmp_system_z[1].subs([(t, 0)])
    tmp_system_z[2] = tmp_system_z[2].subs([(t, 0)])
    print(tmp_system_z)
    lineq = []
    for j in tmp_system_z:
        lineq.append(sympify(str(j)[10:-1]))
    print(lineq)

    res = sp.solve([lineq[0], lineq[1], lineq[2]], (C1, C2, C3), dict=True)
    tmp_system_z[0] = tmp_system_z[0].subs([(C1, 0)])
    tmp_system_z[1] = tmp_system_z[1].subs([(t, 0)])
    tmp_system_z[2] = tmp_system_z[2].subs([(t, 0)])
    print(res)
    PHI = Matrix([[t, 0],
                  [0, t]])

    print(PHI)

    #кінцева точка
    x_T = Matrix([20, 20])

    #початкова точка
    x_0 = Matrix([0, 0])

# знаходимо шукане керування
# обернену матрицю шукаємо за допомогою спраженої
U = B.T*Q.T*(PHI.inverse_ADJ())*(x_T-Q*x_0)

P = lambda t: np.array([U[0], U[1]])
# for i in range(len(U)):
#     print("U_{0} = {1}".format(i+1, U[i]))

F1 = lambda x_1, x_2, x_3: -a*x_1 - b*x_2
F2 = lambda x_1, x_2, x_3: -b*x_1 + g*x_2
F3 = lambda x_1, x_2, x_3: a*x_1 + g*x_2

def system(x, t, a, b, g):
    x_1, x_2, x_3 = x
    dxdt = [F1(x_1, x_2, x_3) + P(t)[0],
            F2(x_1, x_2, x_3) + P(t)[1],
            F3(x_1, x_2, x_3)]
    return dxdt

t0 = 0.
T = 20.
N = 100
time = linspace(t0, T, N + 1)
x0 = [0., 0., 0.]
a = 1
b = 1
g = 1
solution = odeint(system, x0, time, args=(a, b, g))

print(solution[:, 0])
print()
print(solution[:, 1])
print()
print(solution[:, 2])

plt.plot(time, solution[:, 0], 'b', label='x1(t)')
plt.plot(time, solution[:, 1], 'g', label='x2(t)')
plt.plot(time, solution[:, 2], 'r', label='x3(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()