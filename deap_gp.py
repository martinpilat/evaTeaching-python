import operator
import math
import random
import os

import numpy
import pandas as pd

from deap import algorithms, base, creator, tools, gp

import utils

POP_SIZE = 100
CX_PROB = 0.8
MUT_PROB = 0.05
MAX_GEN = 40
INPUT = 'inputs/gp1.csv'
OUT_DIR = 'symbreg'
EXP_ID = 'default'
REPEATS = 10

os.makedirs(OUT_DIR, exist_ok=True)

data = pd.read_csv(INPUT, sep=';')
values = data['y'].values
data = data.drop('y', axis=1)
points = data.values

def safediv(x, y):
    if abs(y) > 0.000001:
        return x/y 
    return 0

def logabs(x):
    return math.log(abs(x)) if x != 0 else 0

# create primitives - non-terminals
pset = gp.PrimitiveSet('MAIN', points.shape[1])
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safediv, 2)
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(logabs, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)

# create the types for fitness and individuals
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

# create the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# define the fitness function (log10 of the rmse or 1000 if overflow occurs)
def eval_symb_reg(individual, points, values):
        try:
            func = toolbox.compile(expr=individual)
            sqerrors = [(func(*z) - valx)**2 for z, valx in zip(points, values)]
            return math.sqrt(math.fsum(sqerrors)) / len(points),
        except OverflowError:
            return 1000.0,

# register the selection and genetic operators - tournament selection and, one point crossover and sub-tree mutation
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("evaluate", eval_symb_reg, points=points, values=values)

# set height limits for the trees
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

for rep in range(REPEATS):
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
        
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_fit.register("avg", numpy.mean)
    stats_fit.register("std", numpy.std)
    stats_fit.register("min", numpy.min)
    stats_fit.register("max", numpy.max)

    stats_size = tools.Statistics(len)
    stats_size.register("avg", numpy.mean)
    stats_size.register("std", numpy.std)
    stats_size.register("min", numpy.min)
    stats_size.register("max", numpy.max)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB, ngen=MAX_GEN, stats=mstats, halloffame=hof, verbose=True)
    pdlog = pd.DataFrame(data={'min': log.chapters['fitness'].select('min'), 
                            'avg': log.chapters['fitness'].select('avg'), 
                            'max': log.chapters['fitness'].select('max')}, 
                        index=numpy.cumsum(log.chapters['fitness'].select('nevals')))
    pdlog.to_csv(f'{OUT_DIR}/{EXP_ID}_{rep}.fitness', header=False, sep=' ')
    pdlog.to_csv(f'{OUT_DIR}/{EXP_ID}_{rep}.objective', header=False, sep=' ')

utils.summarize_experiment(OUT_DIR, EXP_ID)