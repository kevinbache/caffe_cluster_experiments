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
                              "lrdecay=${lr_decay}-${stepsize}_" \
                              "mom=${momentum}_" \
                              "nepochs=${n_epochs}" \
                              ")"
algorithm_template = NamedTemplate(algorithm_name_template_str, algorithm_yaml_template_str)

##########
# params #
##########
cross_params = {
    'train_batch_size': [256],
    # 'learning_rate': np.logspace(-1, -3, 7),  # spacing of 2.15x
    'learning_rate': [0.01],  # spacing of 2.15x
    'lr_decay': [.1],
    'lr_policy': ['step'],
    'stepsize': [100000],
    'momentum': [0.9],
    'seed': np.arange(1)
}
priority = 1
hyper_params = append_dicts(hyper_params, cross_dict(cross_params))

# test_iter: 1000
# test_interval: 1000
# base_lr: 0.01
# lr_policy: "step"
# gamma: 0.1
# stepsize: 100000
# display: 20
# max_iter: 450000
# momentum: 0.9
# weight_decay: 0.0005
# snapshot: 10000



##################
# run experiment #
##################
e = Experiment(use_sge=True, DEBUG_MODE=False)
e.run(experiment_base_name, problem_template, algorithm_template, hyper_params,
      offer_compatible_runs=False, priority=priority)
