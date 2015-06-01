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
problem_file = os.path.join(base_dir, 'problem_template.prototxt')
with open(problem_file, 'r') as f:
    problem_yaml_template_str = f.read()

problem_name_template_str = "ImagenetCaffeNet(wd=${weight_decay}_wf=${weight_fill_name})-" \
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
        if 'train_batch_size' not in hpd:
            for k, v in hpd.items():
                print '%s: %s' % (k, v)
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

experiment_base_name = 'ImagenetVs10'

DRY_RUN = False

hyper_params = {
    # params ends up in run name
    # tag ends up in the problem name
    'params': location,
    'tag': experiment_base_name,

    'n_data_train': 1281167,
    'n_epochs_before_each_snapshot': 1,

    'weight_decay': .0005,

    'n_data_test': 50000,
    'test_batch_size': 50,

    'n_epochs': 90,

    'weight_filler_conv': 'type: "gaussian"\n      std: 0.01',
    'weight_filler_ip': 'type: "gaussian"\n      std: 0.005',
    'weight_filler_ip2': 'type: "gaussian"\n      std: 0.01',

    'shared_cross_params': {
        'train_batch_size': [64, 32, 128, 256],
        'seed': range(1),
    },

    'data_dir': '/data',
    # 'cifar_data': '/data/cifar10/caffe',
    # 'cifar_data': '/storage/code/caffe/examples/cifar10',

    # 'n_max_iters': 1000000,  # will override n_epochs
    'max_seconds': 96 * 3600, # will override n_max_iters
}

def param_extender(hyper_param_dict):
    # fill this function with whatever you'd like.
    # it's a general mechanism for changing hyper parameter sets after the cross params have had
    # their cross products taken
    if 'weight_filler' in hyper_param_dict:
        wf = hyper_param_dict['weight_filler']
    elif 'weight_filler_conv' in hyper_param_dict:
        wf = hyper_param_dict['weight_filler_conv']
    else:
        raise ValueError('Expecting to find either "weight_filler" or "weight_filler_conv" in'
                         'hyper_param_dict')
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


