from __future__ import division
import os
import sys
import socket

"""
Parameters that are shared by all optimizers in this experiment
"""

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

from experiment import NamedTemplate

######################
# load problem yamls #
######################
problem_file = os.path.join(this_path, 'problem_template.prototxt')
with open(problem_file, 'r') as f:
    problem_yaml_template_str = f.read()

problem_name_template_str = "MNIST-FF(${n_neurons_h0}-${n_neurons_h1})-" \
                            "tag(${tag})"
problem_template = NamedTemplate(problem_name_template_str, problem_yaml_template_str)

##############
# parameters #
##############
hostname = socket.gethostname().lower()
if hostname == 'kevins-macbook':
    use_sge = False
    location = 'kevin-macbook'
elif hostname == 'master':
    use_sge = True
    location = 'g2.2xlarge'
else:
    raise ValueError('unknown hostname: %s.  Not sure whether to use Sun Grid Engine.' % hostname)

n_neurons_h0 = 500
n_neurons_h1 = 300

experiment_base_name = 'TEST-EXPERIMENT'

hyper_params = {
    # params ends up in run name
    # tag ends up in the problem name
    'params': location,
    'tag': experiment_base_name,

    # # 'unit_type': 'Sigmoid',
    # 'unit_type': 'RectifiedLinear',

    'n_neurons_h0': n_neurons_h0,
    'n_neurons_h1': n_neurons_h1,

    'n_data_train': 60000,
    'n_data_test': 10000,
    'n_epochs_before_each_snapshot': 10,
    'n_epochs': 200,

    'n_neurons_h0_sparse_init': int(n_neurons_h0 / 10),
    'n_neurons_h1_sparse_init': int(n_neurons_h1 / 10),
    'n_neurons_y_sparse_init': 0,

    # 'h0_bias': 0.,
    # 'h1_bias': 0.,
    #
    # 'col_norm': max_col_norm,
    # 'h0_col_norm': max_col_norm,
    # 'h1_col_norm': max_col_norm,
    # 'y_col_norm': max_col_norm,
    #
    # 'use_dropout': USE_DROPOUT,
}


