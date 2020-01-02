import copy
import csv
import functools
import random

from collections import namedtuple, defaultdict

import numpy as np

import utils

POP_SIZE = 100 # population size
MAX_GEN = 50 # maximum number of generations
CX_PROB = 0.8 # crossover probability
MAX_RULES = 10 # maximum number of rules in an individual
MUT_CLS_PROB = 0.2 # probability of class changing mutation
MUT_CLS_PROB_CHANGE = 0.1 # probability of changing target class in mutation
MUT_COND_PROB = 0.2 # probabilty of condition changing mutation
MUT_COND_SIGMA = 0.3 # step size of condition changing mutation
REPEATS = 10 # number of runs of algorithm (should be at least 10)
INPUT_FILE = 'iris.csv' # the input file for classification
OUT_DIR = 'rules' # output directory for logs
EXP_ID = 'default' # the ID of this experiment (used to create log names)

# a rule is a list of conditions (one for each attribute) and the predicted class
Rule = namedtuple('Rule', ['conditions', 'cls'])

# the following 3 classes implement simple conditions, the call method is used 
# to match the condition against a value
class LessThen:

    def __init__(self, threshold, lb, ub):
        self.params = np.array([threshold])
        self.lb = lb
        self.ub = ub

    def boundary(self):
        return (self.ub - self.lb)*self.params[0] + self.lb

    def __call__(self, value):
        return value <= self.boundary()
    
    def __str__(self):
        return " <= " + str(self.boundary())

class GreaterThen:

    def __init__(self, threshold, lb, ub):
        self.params = np.array([threshold])
        self.lb = lb
        self.ub = ub

    def boundary(self):
        return (self.ub - self.lb)*self.params[0] + self.lb

    def __call__(self, value):
        return value >= self.boundary()

    def __str__(self):
        return " >= " + str(self.boundary())

class Any:

    def __init__(self):
        self.params = np.array([])
    
    def __call__(self, value):
        return True

    def __str__(self):
        return " * "

# generate a single random rule - defines the probabilities of different 
# conditions in the initial population
def create_rule(num_attrs, num_classes, lb, ub):
    conditions = []
    for i in range(num_attrs):
        r = random.random()
        if r < 0.25:
            conditions.append(LessThen(random.random(), lb[i], ub[i]))
        elif r < 0.5:
            conditions.append(GreaterThen(random.random(), lb[i], ub[i]))
        else:
            conditions.append(Any())
    
    return Rule(conditions=conditions, cls=random.randrange(0, num_classes))

# creates the individual - list of rules
def create_ind(max_rules, num_attrs, num_classes, lb, ub):
    ind_len = random.randrange(1, MAX_RULES)
    return [create_rule(num_attrs, num_classes, lb, ub) for i in range(ind_len)]

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# uses an individual to predict a single instance - the rules in the individual
# vote for the final class
def classify_instance(ind, attrs):
    votes = defaultdict(int)
    for rule in ind:
        if all([cond(a) for cond, a in zip(rule.conditions, attrs)]):
            votes[rule.cls] += 1
    
    best_class = None
    best_votes = -1
    for k, v in votes.items():
        if v > best_votes:
            best_votes = v
            best_class = k

    if best_class == None:
        best_class = 0

    return best_class

# computes the accuracy of the individual on a given dataset
def accuracy(ind, data):
    data_x, data_y = data

    correct = 0
    for attrs, target in zip(data_x, data_y):
        if classify_instance(ind, attrs) == target:
            correct += 1
    
    return correct/len(data_y)

# computes the fitness (accuracy on training data) and objective (error rate
# on testing data)
def fitness(ind, train_data, test_data):
    return utils.FitObjPair(fitness=accuracy(ind, train_data), 
                            objective=1-accuracy(ind, test_data))


# the tournament selection
def tournament_selection(pop, fits, k):
    selected = []
    for _ in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if fits[p1] > fits[p2]:
            selected.append(copy.deepcopy(pop[p1]))
        else:
            selected.append(copy.deepcopy(pop[p2]))

    return selected

# implements a uniform crossover for individuals with different lenghts
def cross(p1, p2):
    o1, o2 = [], []
    for r1, r2 in zip(p1, p2):
        if random.random() < 0.5:
            o1.append(copy.deepcopy(r1))
            o2.append(copy.deepcopy(r2))
        else:
            o1.append(copy.deepcopy(r2))
            o2.append(copy.deepcopy(r1))
   
    # individuals can have different lenghts
    l = min(len(p1), len(p2))
    rest = o1[l:] + o2[l:]
    for r in rest:
        if random.random() < 0.5:
            o1.append(copy.deepcopy(r))
        else:
            o2.append(copy.deepcopy(r))

    return o1, o2

# class mutation - changes the predicted class for a given rule
def cls_mutate(p, num_classes):
    p = copy.deepcopy(p)
    o = []
    for r in p:
        o_cls = r.cls
        if random.random() < MUT_CLS_PROB_CHANGE:
            o_cls = random.randrange(0, num_classes)   
        o.append(Rule(conditions=r.conditions, cls=o_cls))
    return o

# mutation changing the threshold in conditions in an individual
def cond_mutate(p):
    o = copy.deepcopy(p)
    for r in o:
        for c in r.conditions:
            c.params += MUT_COND_SIGMA*np.random.randn(*c.params.shape)

    return o

# applies a list of genetic operators (functions with 1 argument - population) 
# to the population
def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off

# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)
def mutation(pop, mutate, mut_prob):
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]

# reads data in a csv file
def read_data(filename):
    data_x = []
    data_y = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            attrs = line[:-1]
            target = line[-1]
            data_x.append(list(map(float, attrs)))
            data_y.append(int(target))

    return (np.array(data_x), np.array(data_y))

# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   operators - list of genetic operators (functions with one arguments - 
#               population; returning a population)
#   mate_sel  - mating selection (funtion with three arguments - population, 
#               fitness values, number of individuals to select; returning the 
#               selected population)
#   mutate_ind - reference to the class to mutate an individual - can be used to 
#               change the mutation step adaptively
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)
        pop = offspring[1:] + [pop[max(enumerate(fits), key=lambda x: x[1])[0]]]

    return pop

if __name__ == '__main__':

    # read the data
    data = read_data('inputs/' + INPUT_FILE)

    num_attrs = len(data[0][0])
    num_classes = max(data[1]) + 1

    # make training and testing split
    perm = np.arange(len(data[1]))
    np.random.shuffle(perm)
    n_train = 2*len(data[1])//3

    train_x, test_x = data[0][perm[:n_train]], data[0][perm[n_train:]]
    train_y, test_y = data[1][perm[:n_train]], data[1][perm[n_train:]]

    # count the lower and upper bounds
    lb = np.min(train_x, axis=0)
    ub = np.max(train_x, axis=0)

    train_data = (train_x, train_y)
    test_data = (test_x, test_y)

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, max_rules=MAX_RULES, 
                               num_attrs=num_attrs, num_classes=num_classes,
                               lb=lb, ub=ub)
    xover = functools.partial(crossover, cross=cross, cx_prob=CX_PROB)
    cls_mutate = functools.partial(cls_mutate, num_classes=num_classes)
    mut_cls = functools.partial(mutation, mutate=cls_mutate, mut_prob=MUT_CLS_PROB)
    mut_cond = functools.partial(mutation, mutate=cond_mutate, mut_prob=MUT_COND_PROB)
    fit = functools.partial(fitness, train_data=train_data, test_data=test_data)

    # run the algorithm `REPEATS` times and remember the best solutions from 
    # last generations

    import multiprocessing

    pool = multiprocessing.Pool(8)
    map_fn = pool.map

    best_inds = []
    for run in range(REPEATS):
        # initialize the log structure
        log = utils.Log(OUT_DIR, EXP_ID, run, write_immediately=True, print_frequency=1)
        # create population
        pop = create_pop(POP_SIZE, cr_ind)
        # run evolution - notice we use the pool.map as the map_fn
        pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut_cls, mut_cond], 
                                     tournament_selection, map_fn=map_fn, log=log)
        # remember the best individual from last generation, save it to file
        bi = max(pop, key=fit)
        best_inds.append(bi)
        
        # if we used write_immediately = False, we would need to save the 
        # files now
        # log.write_files()

    # print an overview of the best individuals from each run
    for i, bi in enumerate(best_inds):
        print(f'Run {i}: objective = {fit(bi).objective}')

    # write summary logs for the whole experiment
    utils.summarize_experiment(OUT_DIR, EXP_ID)