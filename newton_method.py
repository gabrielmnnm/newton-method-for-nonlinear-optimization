import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class Function:
    def __init__(self):
        self.x1, self.x2 = sp.symbols("x1 x2")
        self.f = (self.x1 - 1)**2 + 2 * (2 * self.x2**2 - self.x1)**2
        # self.f = self.x1 - self.x2 + 2 * (self.x1**2) + 2 * self.x1 * self.x2 + (self.x2**2)

    #Vetor gradiente simbolico
    def sym_partial_derivatives(self):
        delf_delx = sp.diff(self.f, self.x1)
        delf_dely = sp.diff(self.f, self.x2)
        return delf_delx, delf_dely

    #Matriz hessiana simbolica
    def sym_second_partial_derivatives(self):
        del2f_delx2 = sp.diff(self.f, self.x1, 2)
        del2f_delxy = sp.diff(self.f, self.x1, self.x2)
        del2f_dely2 = sp.diff(self.f, self.x2, 2)
        del2f_delyx = sp.diff(self.f, self.x2, self.x1)
        return del2f_delx2, del2f_delxy, del2f_dely2, del2f_delyx

    #Funcao numerica (para que possam ser feitas as operacoes matematicas)
    def numeric_function(self, x1, x2):
        numeric_f = sp.lambdify((self.x1, self.x2), self.f, modules='numpy')
        return numeric_f(x1, x2)

    #Vetor gradiente numérico
    def num_partial_derivatives(self, x1, x2):
        delf_delx, delf_dely = self.sym_partial_derivatives()
        numeric_delf_delx = sp.lambdify((self.x1, self.x2), delf_delx, modules='numpy')
        numeric_delf_dely = sp.lambdify((self.x1, self.x2), delf_dely, modules='numpy')
        return numeric_delf_delx(x1, x2), numeric_delf_dely(x1, x2)
    
    #Matriz hessiana numérica
    def num_second_partial_derivatives(self, x1, x2):
        del2f_delx2, del2f_delxy, del2f_dely2, del2f_delyx = self.sym_second_partial_derivatives()
        numeric_del2f_delx2 = sp.lambdify((self.x1, self.x2), del2f_delx2, modules='numpy')
        numeric_del2f_delxy = sp.lambdify((self.x1, self.x2), del2f_delxy, modules='numpy')
        numeric_del2f_dely2 = sp.lambdify((self.x1, self.x2), del2f_dely2, modules='numpy')
        numeric_del2f_delyx = sp.lambdify((self.x1, self.x2), del2f_delyx, modules='numpy')
        return numeric_del2f_delx2(x1, x2), numeric_del2f_delxy(x1, x2), numeric_del2f_dely2(x1, x2), numeric_del2f_delyx(x1, x2)

class Matrix_operations:
    def __init__(self, func: Function):
        self.func = func

    #create a methods to build the gradient and hessian from here
    #create a test to see if the matrix determinant is different than 0 (it must be)
    def num_inverse_matrix(self, x1, x2):
        a11, a12, a21, a22 = self.func.num_second_partial_derivatives(x1, x2)
        A = np.array([[a11, a12], [a21, a22]])
        A_inverse = np.linalg.inv(A)
        return A_inverse

class Direction:
    def __init__(self, func: Function, matr: Matrix_operations):
        self.func = func
        self.matr = matr

    def direction(self, x1, x2):
        A_inverse = self.matr.num_inverse_matrix(x1, x2)

        grad_x, grad_y = self.func.num_partial_derivatives(x1, x2)
        grad = np.array([grad_x, grad_y])

        dir_x, dir_y = A_inverse @ grad
        return -dir_x, -dir_y

class New_point:
    def __init__(self, func: Function, direc: Direction):
        self.func = func
        self.direc = direc

    def new_point(self, x1, x2):
        dir_x, dir_y = self.direc.direction(x1, x2)
        new_x1, new_x2 = x1 + dir_x, x2 + dir_y
        return new_x1, new_x2

func = Function()
matr = Matrix_operations(func)
direc = Direction(func, matr)
npt = New_point(func, direc)

x1, x2 = 0, 0
grad_x, grad_y = func.num_partial_derivatives(x1, x2)
print(grad_x, grad_y)

a11, a12, a21, a22 = func.num_second_partial_derivatives(x1, x2)
A = np.array([[a11, a12], [a21, a22]])
res2 = np.linalg.det(A)
print(res2)

iterations = 0 

# if grad_x == 0 and grad_y == 0:
#     print("Already on a minimum")

# else:
#     while grad_x or grad_y != 0:
#         new_x1, new_x2 = npt.new_point(x1, x2)
#         print(new_x1, new_x2)
#         grad_x, grad_y = func.num_partial_derivatives(new_x1, new_x2)
#         print(grad_x, grad_y)
#         iterations = iterations + 1


#     if iterations == 1:
#         print(f"Sucessfully ended with {iterations} iteration\nNewton's method works for this function")
#     else:
#         print(f"Ended with {iterations} iterations")


##TESTES

# delf_delx, delf_dely = func.sym_partial_derivatives()
# print(delf_delx)
# print(delf_dely)

# print(" ")

# numeric_delf_delx, numeric_delf_dely = func.num_partial_derivatives(x1, x2)
# print(numeric_delf_delx)
# print(numeric_delf_dely)

# print(" ")

# del2f_delx2, del2f_delxy, del2f_dely2, del2f_delyx = func.sym_second_partial_derivatives()
# print(del2f_delx2)
# print(del2f_delxy)
# print(del2f_dely2)
# print(del2f_delyx)

# print(" ")

# numeric_del2f_delx2, numeric_del2f_delxy, numeric_del2f_dely2, numeric_del2f_delyx = func.num_second_partial_derivatives(x1, x2)
# print(numeric_del2f_delx2)
# print(numeric_del2f_delxy)
# print(numeric_del2f_dely2)
# print(numeric_del2f_delyx)

##########################

# A = np.array([[4, 2], [2, 2]])
# res2 = np.linalg.det(A)
# print(res2)

# A = np.array([[50, 29, 56], [77, 30, 44], [15, 63, 90]])
# sign, logdet = np.linalg.slogdet(A)
# res = sign * np.exp(logdet)
# print(res)
