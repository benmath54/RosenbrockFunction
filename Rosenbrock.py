# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:57:45 2024

@author: Ben
"""


import numpy as np
from scipy.optimize import minimize

def Rosenbrock_2D(x):
    a = 1
    b = 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def Rosenbrock_2D_grad(x):
    a = 1
    b = 100
    df_dx0 = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    df_dx1 = 2 * b * (x[1] - x[0]**2)
    return np.array([df_dx0, df_dx1])

def Rosenbrock_4D(x):
    return sum((100*x[i+1]-x[i]**2)**2+(1-x[i])**2 for i in range(3))

def Rosenbrock_4D_grad(x):
    grad = np.zeros_like(x)
    for i in range(3):
        grad[i] = -400 * (x[i + 1] - x[i]**2) * x[i] - 2 * (1 - x[i])
    for i in range(1, 4):
        grad[i] += 200 * (x[i] - x[i - 1]**2)
    return grad

deg = int(input("2d or 4d: "))
x0 = []
if(deg == 2):
    for i  in range(0,deg):
        temp = int(input("guess: "))
        x0.append(temp)
    result = minimize(Rosenbrock_2D, x0,jac=Rosenbrock_2D_grad)
    print("Optimal parameters:", result.x)
    print("Minimum value:", result.fun)
if(deg == 4):
    for i  in range(0,deg):
        temp = int(input("guess: "))
        x0.append(temp)
    result = minimize(Rosenbrock_2D, x0,jac=Rosenbrock_2D_grad)
    print("Optimal parameters:", result.x)
    print("Minimum value:", result.fun)    