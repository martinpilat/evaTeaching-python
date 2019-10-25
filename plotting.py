# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt 

plt.figure(figsize=(12,8))
utils.plot_experiments('continuous', ['default.f01', 'tuned.f01'], 
                       rename_dict={'default.f01': 'Sphere (default)',
                                    'tuned.f01': 'Sphere (tuned)'})
plt.show()
 