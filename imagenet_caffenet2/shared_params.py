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

problem_name_template_str = "CifarAlexCaffe-" \
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

experiment_base_name = 'TEST-EXPERIMENT'

hyper_params = {
    # params ends up in run name
    # tag ends up in the problem name
    'params': location,
    'tag': experiment_base_name,

    'imagenet_lmdbs_dir': '/storage/data/imagenet',

    'n_data_train': 1281167,
    'n_epochs_before_each_snapshot': 1,

    'n_data_test': 50000,
    'test_batch_size': 50,

    'n_iters_before_display': 20,

    'n_epochs': 90,
}

# net: "models/bvlc_reference_caffenet/train_val.prototxt"
# test_iter: 1000
# test_interval: 1000
# display: 20
# max_iter: 450000
# snapshot: 10000
# snapshot_prefix: "models/bvlc_reference_caffenet/caffenet_train"
# solver_mode: GPU

