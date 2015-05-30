from __future__ import division
import os
import sys
import numpy as np

##################################
# set up paths and extra imports #
##################################
def parent(dirname, n_levels):
    for _ in range(n_levels):
        dirname = os.path.dirname(dirname)
    return dirname

this_path = os.path.dirname(os.path.realpath(__file__))
base_dir = parent(this_path, 1)
sys.path.append(base_dir)
sys.path.append(this_path)

from experiment import Experiment, NamedTemplate, append_dicts, cross_dict
from shared_params import *

####################
# set up templates #
####################
with open(os.path.join(solvers_dir, 'solver_ad_template.prototxt'), 'r') as f:
    algorithm_yaml_template_str = f.read()

algorithm_name_template_str = "AD(" \
                              "batch=${train_batch_size}_" \
                              "mom=${momentum}_" \
                              "delta=${delta}_" \
                              "nepochs=${n_epochs}" \
                              ")"
algorithm_template = NamedTemplate(algorithm_name_template_str, algorithm_yaml_template_str)

##########
# params #
##########
all_cross_params = hyper_params['shared_cross_params']
del hyper_params['shared_cross_params']
n_shared_cross = count_cross_possibilities(all_cross_params)
cross_params = {
    'momentum': [0.95],
    'delta': [1e-8],
}
all_cross_params.update(cross_params)
priority = 3
hyper_param_dicts = append_dicts(hyper_params, cross_dict(all_cross_params))

# extend corssed params with extra values (e.g. 'weight_filler_name')
hyper_param_dicts = [param_extender(hpd) for hpd in hyper_param_dicts]


##################
# run experiment #
##################
if DRY_RUN:
    print_hyper_param_dicts(n_shared_cross, all_cross_params, hyper_param_dicts, algorithm_template)
    print 'Exiting'
    exit()

e = Experiment(use_sge=True, DEBUG_MODE=False)
e.run(experiment_base_name, problem_template, algorithm_template, hyper_param_dicts,
      offer_compatible_runs=False, priority=priority)
