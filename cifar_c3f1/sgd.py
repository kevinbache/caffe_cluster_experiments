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
with open(os.path.join(this_path, 'solver_sgd_template.prototxt'), 'r') as f:
    algorithm_yaml_template_str = f.read()

algorithm_name_template_str = "SGD(" \
                              "batch=${train_batch_size}_" \
                              "lr=${learning_rate}_" \
                              "lrdecay=${lr_decay}${stepsize}_" \
                              "mom=${momentum}_" \
                              "nepochs=${n_epochs}" \
                              ")"
algorithm_template = NamedTemplate(algorithm_name_template_str, algorithm_yaml_template_str)

##########
# params #
##########
cross_params = {
    'train_batch_size': [125],
    # 'train_batch_size': [50, 250, 500],
    'learning_rate': np.logspace(-1, -4, 13),  # spacing of 1.77x
    'lr_decay': [.95, .96],
    'lr_policy': ['step'],
    'stepsize': [600],
    'momentum': [0.0],
    'seed': np.arange(1)
}
priority = 1
hyper_params = append_dicts(hyper_params, cross_dict(cross_params))

##################
# run experiment #
##################
e = Experiment(use_sge=True, DEBUG_MODE=False)
e.run(experiment_base_name, problem_template, algorithm_template, hyper_params,
      offer_compatible_runs=False, priority=priority)
