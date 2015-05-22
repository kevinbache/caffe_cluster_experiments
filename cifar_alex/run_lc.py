from __future__ import division
import os
import sys
import numpy as np

################
# set up paths #
################
def parent(dirname, n_levels):
    for _ in range(n_levels):
        dirname = os.path.dirname(dirname)
    return dirname

this_path = os.path.dirname(os.path.realpath(__file__))
base_dir = parent(this_path, 1)
sys.path.append(base_dir)
sys.path.append(this_path)

##########################
# path dependent imports #
##########################
from experiment import Experiment, NamedTemplate, append_dicts, cross_dict
from shared_params import *

#############
# load yaml #
#############
with open(os.path.join(this_path, 'solver_lc_template.prototxt'), 'r') as f:
    algorithm_yaml_template_str = f.read()

####################
# set up templates #
####################
algorithm_name_template_str = "LC(" \
                              "batch=${batch_size}_" \
                              "min=${log_alpha_min}_" \
                              "max=${log_alpha_max}_" \
                              "n=${n_alphas}_" \
                              "nepochs=${n_epochs}" \
                              ")"
algorithm_template = NamedTemplate(algorithm_name_template_str, algorithm_yaml_template_str)

##########
# params #
##########
cross_params = {
    'batch_size': [25, 50, 125, 250, 500, 1000],
    'log_alpha_min': [-6],
    'log_alpha_max': [8],
    'n_alphas': [99],
    'seed': np.arange(3)
}
priority = 0
hyper_params = append_dicts(hyper_params, cross_dict(cross_params))

##################
# run experiment #
##################
e = Experiment(use_sge=True, DEBUG_MODE=False)
e.run(experiment_base_name, problem_template, algorithm_template, hyper_params,
      offer_compatible_runs=False, priority=priority)
