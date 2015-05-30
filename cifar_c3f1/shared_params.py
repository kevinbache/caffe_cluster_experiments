from __future__ import division
import os
import re
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

solvers_dir = os.path.join(base_dir, 'solver_templates')

from experiment import NamedTemplate

######################
# load problem yamls #
######################
problem_file = os.path.join(this_path, 'problem_template.prototxt')
with open(problem_file, 'r') as f:
    problem_yaml_template_str = f.read()

problem_name_template_str = "CifarAlexCaffe(wd=${weight_decay}_wf=${weight_fill_name})-" \
                            "tag(${tag})"
problem_template = NamedTemplate(problem_name_template_str, problem_yaml_template_str)

#############
# functions #
#############
def count_cross_possibilities(cross_param_dict):
    import numpy as np
    return np.prod([len(v) for v in cross_param_dict.values()])

def print_hyper_param_dicts(n_shared_cross, cross_param_dict, hyper_param_dicts, alg_template):
    for hpd in hyper_param_dicts:
        print problem_template.fill_name(hpd), alg_template.fill_name(hpd)
        for k, v in hpd.items():
            print '%s: %s' % (k, v)
            print
            print 

    n_total_cross = count_cross_possibilities(cross_param_dict)
    print '======================================================'
    print '%d shared, %d local, %d hpds total' % (n_shared_cross,
                                                  int(n_total_cross / n_shared_cross),
                                                  int(n_total_cross))


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

experiment_base_name = 'CifarVs5'

DRY_RUN = True

hyper_params = {
    # params ends up in run name
    # tag ends up in the problem name
    'params': location,
    'tag': experiment_base_name,

    'n_data_train': 50000,
    'n_data_test': 10000,
    'n_epochs_before_each_snapshot': 10,
    'n_epochs': 1000,

    'test_batch_size': 250,

    'weight_decay': .0005,

    'shared_cross_params': {
        'weight_filler': ['      type: "gaussian"\n      std: 0.001\n',
                          '      type: "xavier"\n'],
        'train_batch_size': [50, 80, 125, 250],
        'seed': range(2),
    },


    'n_max_iters': 1000000,  # will override n_epochs
    'max_seconds': 4 * 3600, # will override n_max_iters
}

def param_extender(hyper_param_dict):
    # fill this function with whatever you'd like.
    # it's a general mechanism for changing hyper parameter sets after the cross params have had
    # their cross products taken

    wf = hyper_param_dict['weight_filler']
    if 'gauss' in wf:
        m = re.search(r'std: ([\.\d\-+e]+)', wf)
        std = float(m.group(1))
        name = 'gauss-%f' % std
    elif 'xavier' in wf:
        name = 'xavier'
    else:
        raise ValueError('Unknown weight filler')
    hyper_param_dict['weight_fill_name'] = name

    return hyper_param_dict


