import random
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import utils

K = 10

# reads the input set of values of objects
def read_weights(filename):
    with open(filename) as f:
        return list(map(int, f.readlines()))

# computes the bin weights
# - bins are the indices of bins into which the object belongs
def bin_weights(weights, bins):
    bw = [0]*K
    for w, b in zip(weights, bins):
        bw[b] += w
    return bw

# the fitness function
def fitness(ind, weights):
    bw = bin_weights(weights, ind)
    return max(bw) - min(bw),

weights = read_weights('inputs/partition-easy.txt')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 9)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(weights))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



toolbox.register("evaluate", fitness, weights=weights)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=9, indpb=0.005)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.8, ngen=100, stats=stats, halloffame=hof, verbose=True)
    
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()
    print('Best individual fitness:', hof[0].fitness.values)

