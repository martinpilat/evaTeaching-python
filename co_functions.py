import numpy as np
import functools
from utils import FitObjPair
import pprint

# this file implements some of the functions from the BBOB benchmark, see 
# http://coco.lri.fr/downloads/download15.02/bbobdocfunctions.pdf for details

# generates random rotation matrix of size n
def random_rotation_matrix(n):
    R = np.random.normal(size=(n,n))
    R, _ = np.linalg.qr(R)
    return R

# transformation used in some of the functions
def Lambda(alpha, D):
    l = np.power(np.full(D, alpha), 0.5*np.arange(0, D)/(D - 1))
    return np.diag(l)

# another transformation used in some of the functions
def T_osz(x):
    c1 = np.full(x.shape, 10.0)
    c2 = np.full(x.shape, 7.9)
    c1[x<=0] = 5.5
    c2[x<=0] = 3.1
    x_hat = np.log(np.abs(x))
    return np.sign(x)*np.exp(x_hat + 0.049*(np.sin(c1*x_hat) + np.sin(c2*x_hat)))

# generates random value of the optimum
def random_fopt():
    return np.clip(np.round(100*np.random.normal()/np.random.normal(), 2), -1000, 1000)

# generates random location for the optimum
def random_xopt(n):
    return np.random.uniform(-4, 4, size=(n,))

# bellow are implementations of selected functions from the benchmark, the 
# implementation is divided into two parts - the function itself and a function
# that creates its instance (all the funtions are randomized to hide the optimum
# value and the location of the optimum from the algorithm)
# the objective value returned by the function represents how far the solution 
# is from the optimum one in terms of function values


def f01_sphere(x, xopt, fopt):
    z = x - xopt
    val = np.linalg.norm(z)**2 + fopt
    return FitObjPair(fitness = -val, objective = val - fopt)

def make_f01_sphere(dim):
    xopt = random_xopt(dim)
    fopt = random_fopt()
    return functools.partial(f01_sphere, xopt = xopt, fopt = fopt)

def f02_ellipsoidal(x, xopt, fopt, D, i):
    z = T_osz(x - xopt)
    val = np.sum(np.power(np.full(D, 10), i/(D-1))*z*z) + fopt
    return FitObjPair(fitness = -val, objective = val - fopt)

def make_f02_ellipsoidal(dim):
    xopt = random_xopt(dim)
    fopt = random_fopt()
    i = np.arange(0, dim)
    return functools.partial(f02_ellipsoidal, xopt = xopt, fopt = fopt, 
                             D = dim, i = i)

def f06_attractive_sector(x, xopt, fopt, Q, R, L, D):
    z = Q@L@R@(x-xopt)
    s = np.full((D,), 1)
    s[z*xopt > 0] = 100
    val = np.power(T_osz(np.sum(np.power(s*z, 2))), 0.9) + fopt
    return FitObjPair(fitness = -val, objective = val - fopt)

def make_f06_attractive_sector(dim):
    xopt = random_xopt(dim)
    fopt = random_fopt()
    R = random_rotation_matrix(dim)
    Q = random_rotation_matrix(dim)
    L = Lambda(10, dim)
    return functools.partial(f06_attractive_sector, xopt = xopt, fopt = fopt, 
                             R = R, Q = Q, L = L, D = dim)


def f08_rosenbrock(x, xopt, fopt, D):
    z = np.max((1, np.sqrt(D)/8))*(x - xopt) + 1
    val = np.sum(np.power(np.power(z[:-1], 2) - z[1:], 2) + np.power(z[:-1]-1, 2)) + fopt
    return FitObjPair(fitness = -val, objective = val - fopt)

def make_f08_rosenbrock(dim):
    xopt = random_xopt(dim)
    fopt = random_fopt()
    return functools.partial(f08_rosenbrock, xopt = xopt, fopt = fopt, D = dim)

def f10_rotated_ellipsoidal(x, xopt, fopt, R, D, i):
    z = T_osz(R@(x - xopt))
    val = np.sum(np.power(np.full(D, 10), i/(D-1))*z*z) + fopt
    return FitObjPair(fitness = -val, objective = val - fopt)

def make_f10_rotated_ellipsoidal(dim):
    xopt = random_xopt(dim)
    fopt = random_fopt()
    i = np.arange(0, dim)
    R = random_rotation_matrix(dim)
    return functools.partial(f10_rotated_ellipsoidal, xopt = xopt, fopt = fopt, 
                             R = R, D = dim, i = i)


def numerical_derivative(f, x):
    fx = f(x).fitness
    d = np.zeros_like(x)
    for i in range(x.shape[0]):
        xph = x.copy()
        xph[i] += 0.0000001*x[i]
        dx = xph[i] - x[i]
        d[i] = (f(xph).fitness - fx)/dx
    
    return -d # the fitness is the negative of the function - need to take 
              # negative again (to match the Java implementation)

# bellow is commented code that shows how to use the functions and also can be 
# used to create plots of the functions with 2 variables

# f01 = make_f01_sphere(2)
# f02 = make_f02_ellipsoidal(2)
# f06 = make_f06_attractive_sector(2)
# f08 = make_f08_rosenbrock(2)
# f10 = make_f10_rotated_ellipsoidal(2)
# x = np.random.uniform(-5, 5, size=(2,))
# print(x)
# print(f01(x))
# print(f02(x))
# print(f06(x))
# print(f08(x))
# print(f10(x))

# x = y = np.arange(-5.0, 5.0, 0.05)
# X, Y = np.meshgrid(x, y)
# Z = np.vstack([np.ravel(X), np.ravel(Y)]).T

# zs = []
# for i in range(Z.shape[0]):
#     zs.append(f10(Z[i]).objective)

# import matplotlib.pyplot as plt 
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(12,8))
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(X, Y, np.reshape(zs, X.shape))
# plt.show()

