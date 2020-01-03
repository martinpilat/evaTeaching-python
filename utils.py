from collections import namedtuple
import os
import glob
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import multi_utils

# a tuple of fitness value and objective value
FitObjPair = namedtuple('FitObjPair', ['fitness', 'objective'])

# can plot a single experiment from the statistical information 
# Arguments:
#   evals - number of evalutions
#   lower - lower boundary of shaded area
#   mean - the main line in the plot
#   upper - upper boundary of shaded area
#   legend_name - name in the plot legend
def plot_experiment(evals, lower, mean, upper, legend_name=''):
    plt.plot(evals, mean, label=legend_name)
    plt.fill_between(evals, lower, upper, alpha=0.25)

# reads the run logs and computes experiment statisticts (those used for plots)
# Arguments:
#   prefix - directory with experiments
#   exp_id - name of the experiment
#   stat_type - either 'objective' or 'fitness', used to specify which values 
#               to use for the stats
def get_experiment_stats(prefix, exp_id, stat_type='objective'):
    data = []
    for fn in glob.glob(f'{prefix}/{exp_id}_*.{stat_type}'):
        evals, stats = read_run_file(fn)
        data.append(pd.Series([s.max for s in stats], index=evals))
    data_frame = pd.DataFrame(data)
    data_frame.fillna(method='ffill', inplace=True, axis=1)
    return (evals, np.min(data_frame, axis=0), 
            np.percentile(data_frame, q=25, axis=0), 
            np.mean(data_frame, axis=0), 
            np.percentile(data_frame, q=75, axis=0),
            np.max(data_frame, axis=0))

# the same as get_experiment_stats, but returns only some of the data
def get_plot_data(prefix, exp_id, stat_type='objective'):
    evals, lower, q25, mean, q75, upper = get_experiment_stats(prefix, exp_id, stat_type)
    return evals, q25, mean, q75

# computes the experimets stats for both objective and fitness values and prints
# them to file
def summarize_experiment(prefix, exp_id):
    evals, lower, q25, mean, q75, upper = get_experiment_stats(prefix, exp_id, 'fitness')
    with open(f'{prefix}/{exp_id}.fitness_stats', 'w') as f:
        for e,l,q2,m,q7,u in zip(evals, lower, q25, mean, q75, upper):
            f.write(f'{e} {l} {q2} {m} {q7} {u}\n')
    
    evals, lower, q25, mean, q75, upper = get_experiment_stats(prefix, exp_id, 'objective')
    with open(f'{prefix}/{exp_id}.objective_stats', 'w') as f:
        for e,l,q2,m,q7,u in zip(evals, lower, q25, mean, q75, upper):
            f.write(f'{e} {l} {q2} {m} {q7} {u}\n')

# reads one file with log of a single run (as produced by the Log class)
def read_run_file(filename):
    stats = []
    evals = []
    with open(filename) as f:
        for line in f.readlines():
            e, x, m, n = line.split(' ')
            evals.append(int(e))
            stats.append(GenStats(min=float(n), max=float(x), mean=float(n)))

    return evals, stats

# plots a number of experiments
# Arguments:
#   prefix - directory with experimental results
#   exp_ids - list of experiments to plot
#   rename_dict - a mapping of exp_id -> legend name, can be used to rename 
#                 entries in the plot legend
#   stat_type - either 'objective' or 'fitness' - the type of values to plot
def plot_experiments(prefix, exp_ids, rename_dict=None, stat_type='objective'):
    if not rename_dict:
        rename_dict = dict()
    for e in exp_ids:
        evals, lower, mean, upper = get_plot_data(prefix, e, stat_type)
        plot_experiment(evals, lower, mean, upper, rename_dict.get(e, e))
    plt.legend()
    plt.xlabel('Fitness evaluations')
    if stat_type == 'objective':
        plt.ylabel('Objective value')
    if stat_type == 'fitness':
        plt.ylabel('Fitness value')

# a tuple for the stats about a single generation
GenStats = namedtuple('GenStats', ['min', 'max', 'mean'])

# this class produces the log files/console output
class Log:    

    # Arguments:
    #   prefix - the directory for the logs
    #   exp_id - id of the experiment (cannot contain '_')
    #   run_id - id of the run in the experiment
    #   write_immediately - whether to write the output files immediately 
    #                       or wait till the `write_file` method is called
    #   print_frequency - how often to print the output to console
    #   remove_existing - whether to remove existing files for given combination
    #                     of exp_id and run_id (otherwise new values are 
    #                     appended)
    def __init__(self, prefix, exp_id, run_id, write_immediately = False, 
                 print_frequency = 1, remove_existing = True):
        self.prefix = prefix
        self.exp_id = exp_id
        if '_' in exp_id:
            warnings.warn('Experiment ID should not contain \'_\'')
        self.run_id = run_id
        self.write_immediately = write_immediately
        self.print_frequency = print_frequency
        self.gen_num = 0
        self.gens = []
        self.fit_stats = []
        self.obj_stats = []
        self.fevals = []
        os.makedirs(prefix, exist_ok=True)
        self.flog_name = f'{self.prefix}/{self.exp_id}_{self.run_id}.fitness'
        self.olog_name = f'{self.prefix}/{self.exp_id}_{self.run_id}.objective'
        if remove_existing:
            if os.path.exists(self.flog_name):
                os.remove(self.flog_name)
            if os.path.exists(self.olog_name):
                os.remove(self.olog_name)
    
    # add information about the generation into the log
    def add_gen(self, fit_obj, f_evals):
        self.gen_num += 1
        self.gens.append(fit_obj)
        self.fevals.append(f_evals)

        fits = [f.fitness for f in fit_obj]
        objs = [f.objective for f in fit_obj]

        fs = GenStats(min=min(fits), max=max(fits), mean=sum(fits)/len(fits))
        os = GenStats(min=min(fit_obj, key=lambda x: x.fitness).objective,
                      mean=sum(objs)/len(objs),
                      max=max(fit_obj, key=lambda x: x.fitness).objective)

        self.fit_stats.append(fs)
        self.obj_stats.append(os)

        if self.write_immediately:
            with open(self.flog_name, 'a') as f:
                f.write(f'{f_evals} {fs.max} {fs.mean} {fs.min}\n')
            with open(self.olog_name, 'a') as f:
                f.write(f'{f_evals} {os.max} {os.mean} {os.min}\n')
        
        if self.gen_num % self.print_frequency == 0:
            print(f'{f_evals:8} {os.min:8.2f} {os.mean:8.2f} {os.max:8.2f}')

    def add_multi_gen(self, pop, f_evals, opt_hv):
        self.gen_num += 1
        self.gens.append(None)
        self.fevals.append(f_evals)

        hv = multi_utils.hypervolume(pop)

        os = GenStats(min=opt_hv - hv, max=opt_hv - hv, mean=opt_hv - hv)
        fs = GenStats(min=hv, max=hv, mean=hv)

        self.obj_stats.append(os)
        self.fit_stats.append(fs)

        if self.write_immediately:
            with open(self.flog_name, 'a') as f:
                f.write(f'{f_evals} {fs.max} {fs.mean} {fs.min}\n')
            with open(self.olog_name, 'a') as f:
                f.write(f'{f_evals} {os.max} {os.mean} {os.min}\n')
        
        if self.gen_num % self.print_frequency == 0:
            print(f'{f_evals:8} {os.min:8.2f} {os.mean:8.2f} {os.max:8.2f}')

    def write_files(self):
        with open(self.flog_name, 'a') as f:
            for f_evals, fs in zip(self.fevals, self.fit_stats):
                f.write(f'{f_evals} {fs.max} {fs.mean} {fs.min}\n')
        with open(self.olog_name, 'a') as f:
            for f_evals, os in zip(self.fevals, self.obj_stats):
                f.write(f'{f_evals} {os.max} {os.mean} {os.min}\n')
