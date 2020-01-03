import numpy as np

from math import sqrt, sin, pi, cos, exp

from utils import FitObjPair

# these implementations are based on those from the deap library 
# https://github.com/DEAP/deap/

def get_function_by_name(name):
    if name == 'ZDT1':
        return zdt1
    if name == 'ZDT2':
        return zdt2
    if name == 'ZDT3':
        return zdt3
    if name == 'ZDT4':
        return zdt4
    if name == 'ZDT6':
        return zdt6
    
    raise RuntimeError(f'Specified benchmark function ({name}) not found.')

opt_hvs = {
    'ZDT1': 120.0 + 2/3,
    'ZDT2': 120.0 + 1/3,
    'ZDT3': 128.77811613069076060,
    'ZDT4': 120.0 + 2/3,
    'ZDT6': 117.51857519692037009,
}

def get_opt_hypervolume(name):
    if name in opt_hvs.keys():
        return opt_hvs[name]
    
    raise RuntimeError(f'Specified benchmark function ({name}) not found.')


def zdt1(individual):
    g  = 1.0 + 9.0*sum(individual.x[1:])/(len(individual.x)-1)
    f1 = individual.x[0]
    f2 = g * (1 - sqrt(f1/g))
    return (f1, f2)

def zdt2(individual):
    g  = 1.0 + 9.0*sum(individual.x[1:])/(len(individual.x)-1)
    f1 = individual.x[0]
    f2 = g * (1 - (f1/g)**2)
    return (f1, f2)
    
def zdt3(individual):
    g  = 1.0 + 9.0*sum(individual.x[1:])/(len(individual.x)-1)
    f1 = individual.x[0]
    f2 = g * (1 - sqrt(f1/g) - f1/g * sin(10*pi*f1))
    return (f1, f2)

def zdt4(individual):
    g  = 1 + 10*(len(individual.x)-1) + sum(xi**2 - 10*cos(4*pi*xi) for xi in individual.x[1:])
    f1 = individual.x[0]
    f2 = g * (1 - sqrt(f1/g))
    return (f1, f2)
    
def zdt6(individual):
    g  = 1 + 9 * (sum(individual.x[1:]) / (len(individual.x)-1))**0.25
    f1 = 1 - exp(-4*individual.x[0]) * sin(6*pi*individual.x[0])**6
    f2 = g * (1 - (f1/g)**2)
    return (f1, f2)
