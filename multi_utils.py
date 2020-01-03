import operator
import numpy as np


HYP_REF = (11, 11) # reference point for hypervolume (do not change - results will not be comparable)

def hypervolume(pop, ref=HYP_REF):
    non_dom = [pop[i] for i in get_first_nondominated(pop)]

    f1_sorted = list(sorted(non_dom, key=lambda x: x.fitness[0]))

    volume = 0

    for (i, j) in zip(f1_sorted, f1_sorted[1:]):
        volume += (ref[1] - i.fitness[1])*(j.fitness[0] - i.fitness[0])
    
    volume += (ref[1] - f1_sorted[-1].fitness[1])*(ref[0] - f1_sorted[-1].fitness[0])

    return volume

def assign_crowding_distances(front):
    front = list(sorted(front, key=operator.attrgetter('fitness')))
    front[0].ssc = np.inf # first and last one have infinite crowding distance
    front[-1].ssc = np.inf
    for i in range(1, len(front) - 1):
        front[i].ssc = (front[i + 1].fitness[0] - front[i - 1].fitness[0] +
                        front[i - 1].fitness[1] - front[i + 1].fitness[1])

# returns true if i1 dominates i2
def dominates(fits1, fits2):
    return (all(map(lambda x: x[0] <= x[1], zip(fits1, fits2))) and 
            any(map(lambda x: x[0] < x[1], zip(fits1, fits2))))

def get_first_nondominated(pop):
    non_dom = []
    for i, p in enumerate(pop):
        if not any(map(lambda x: dominates(x.fitness, pop[i].fitness), pop)):
            non_dom.append(i)
    return non_dom

def divide_fronts(pop):
    fronts = []
    while pop:
        non_dom = get_first_nondominated(pop)
        front = [pop[i] for i in non_dom]        
        pop = [p for i,p in enumerate(pop) if i not in non_dom]
        fronts.append(front)
    return fronts