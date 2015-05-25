import glob
import os
import cPickle
from time import localtime, strftime
from string import Template
import subprocess

this_path = os.path.dirname(os.path.realpath(__file__))

def print_named_content(name, content):
    header_len = 100
    print
    print '=' * header_len
    print name
    print '-' * header_len
    print content
    print '=' * header_len
    print


# to my eyes, the """ """s with newlines inside makes it cleaner to keep this outside of the class
sge_template = Template("""
#!/bin/bash
mkdir -p "${tmp_output_path}/"
"${caffe_binary_fullfile}" train --solver="${algorithm_fullfile}"

# move the latest solverstate and caffemodel to the final output path
mv $(ls -t \"${tmp_output_path}\"/*.solverstate | head -1) "${final_output_path}"
mv $(ls -t \"${tmp_output_path}\"/*.caffemodel | head -1) "${final_output_path}"

"""
)

# separators between fields; used to customize path and file name formats
seps = {'minor': '_', 'major': '--', 'super': '----'}

class Experiment(object):
    # leave these as class variables so they can be accessed from TimeSeriesPlotter
    default_final_output_path = os.path.join(this_path, 'output')
    default_tmp_output_path = '/scratch/sgeadmin/output/'  # used if running on sun grid engine
    default_sge_final_output_path = '/storage/output/'

    default_data_path_addon = 'runs'
    default_experiment_path_addon = 'experiments'
    # these will be checked in order to see if the file exists at each location before use
    default_caffe_roots = ['/storage/code/caffe',
                           '/Users/kevin/projects/caffe']
    default_caffe_bin_addon = 'build/tools/caffe'

    # a 'run' represents a single call to train.main_loop()
    # it is a combination of a problem, dataset, training algorithm, and hyper parameters
    run_name_template = Template("${problem_name}"
                                 "${major}"
                                 "${algorithm_name}"
                                 "${super}"
                                 "${params}"
                                 "${major}"
                                 "seed=${seed}")
    experiment_filename_template = Template("${experiment_base_name}${super}${start_time}.txt")

    # name of files that this class saves in each run's directory
    name_problem_file = 'problem.prototxt'
    name_algorithm_file = 'algorithm.prototxt'
    name_names_hyper_params = 'names_and_hyper_params.pkl'
    # name_final_yaml = 'final.yaml'
    name_trained_model = 'trained_model.pkl'
    # a smaller subset of trained_model.pkl for faster loading for plotting
    trained_model_small = 'trained_model_small.pkl'
    channels_filename = 'channels.pkl'

    # year_month_day-hour_minute_second
    time_format = Template("%Y${minor}%m${minor}%d"
                           "${major}"
                           "%H${minor}%M${minor}%S").safe_substitute(seps)

    def __init__(self,
                 use_sge=False,
                 final_output_path=None,
                 data_path=None,
                 experiment_path=None,
                 caffe_paths=None,
                 tmp_output_path=None,
                 DEBUG_MODE=False):

        self.use_sge = use_sge
        self.DEBUG_MODE = DEBUG_MODE

        if final_output_path is None:
            if self.use_sge:
                self.final_output_path = self.default_sge_final_output_path
            else:
                self.final_output_path = self.default_final_output_path

        if data_path is None:
                self.data_path = os.path.join(self.final_output_path, self.default_data_path_addon)

        if experiment_path is None:
            self.experiment_path = os.path.join(self.final_output_path, self.default_experiment_path_addon)

        if caffe_paths is None:
            self.caffe_binary_fullfile = None
            self.caffe_root = None
            for p in self.default_caffe_roots:
                caffe_fullfile = os.path.join(p, self.default_caffe_bin_addon)
                if os.path.isfile(caffe_fullfile):
                    self.caffe_binary_fullfile = caffe_fullfile
                    self.caffe_root = p
                    break
            if self.caffe_binary_fullfile is None:
                raise ValueError('Could not find caffe binary at any of:', self.default_caffe_paths)

        if tmp_output_path is None:
            self.tmp_output_path = self.default_tmp_output_path

        if not self.DEBUG_MODE:
            if self.use_sge:
                self.makedir(self.tmp_output_path)
            self.makedir(self.final_output_path)
            self.makedir(self.data_path)
            self.makedir(self.experiment_path)

    @staticmethod
    def makedir(directory):
        """ create a directory if it doesn't already exist """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_time_str(self):
        return strftime(self.time_format, localtime())

    @staticmethod
    def validate_run_names(run_names):
        if len(run_names) != len(set(run_names)):
            raise ValueError('Run names should all be unique, but run_names = \n' + '\n'.join(run_names))

    @staticmethod
    def add_batch_size_params(hyper_param_dicts):
        required_params = ['n_data_train',
                           'n_data_test',
                           'batch_size',
                           'n_epochs_before_each_snapshot',
                           'n_epochs']
        for hyper_param_dict in hyper_param_dicts:
            if set(required_params) <= set(hyper_param_dict.keys()):
                batch_size = hyper_param_dict['batch_size']
                n_data_train = hyper_param_dict['n_data_train']
                n_data_test = hyper_param_dict['n_data_test']
                assert not n_data_train % batch_size
                assert not n_data_test % batch_size

                n_epochs = hyper_param_dict['n_epochs']
                n_epochs_before_each_snapshot = hyper_param_dict['n_epochs_before_each_snapshot']

                n_iters_per_epoch = int(n_data_train / batch_size)
                n_iters_per_test = int(n_data_test / batch_size)

                d = {
                    'train_batch_size': batch_size,
                    'test_batch_size': batch_size,

                    'n_test_on_train_iters': n_iters_per_epoch,
                    'n_test_on_test_iters': n_iters_per_test,

                    'n_iters_before_display': int(n_iters_per_epoch / 10),
                    'n_iters_before_test': n_iters_per_epoch,
                    'n_max_iters': n_iters_per_epoch * n_epochs,
                    'n_iters_before_snapshot': n_iters_per_epoch * n_epochs_before_each_snapshot,
                }
                hyper_param_dict.update(d)
            else:
                missing_keys = [k for k in required_params if k not in hyper_param_dict]
                raise ValueError("A hyperparameter dict is missing the keys: "
                                 + ', '.join(missing_keys))

    def run(self, experiment_base_name, problem_template, algorithm_template, hyper_param_dicts,
            offer_compatible_runs=True, priority=0):
        if isinstance(hyper_param_dicts, dict):
            hyper_param_dicts = [hyper_param_dicts,]

        self.add_batch_size_params(hyper_param_dicts)

        start_time = self.get_time_str()
        run_names = self.compile_run_names(problem_template, algorithm_template, hyper_param_dicts, start_time)
        self.validate_run_names(run_names)

        if offer_compatible_runs:
            all_compatible_runs = self.find_compatible_runs(run_names)
            use_runs = self.offer_compatible_runs(run_names, all_compatible_runs)
        else:
            all_compatible_runs = [[]] * len(run_names)
            use_runs = [0] * len(run_names)

        used_run_paths = []
        for hyper_param_dict, \
            run_name, \
            use_run, \
            compatible_runs in zip(hyper_param_dicts, run_names, use_runs, all_compatible_runs):

            if use_run != 0:
                # load an existing run instead of performing a new one
                final_output_path = compatible_runs[use_run-1]
            else:
                final_output_path = self.get_this_final_run_path(run_name)
                if self.use_sge:
                    tmp_output_path = self.get_this_tmp_run_path(run_name)
                else:
                    tmp_output_path = final_output_path
                if not self.DEBUG_MODE:
                    self.makedir(final_output_path)
                self.save_names_and_hyper_params(problem_template, algorithm_template, hyper_param_dict, final_output_path)
                self.save_problem_and_algorithm(problem_template, algorithm_template, hyper_param_dict, final_output_path, tmp_output_path)

                # perform a new run
                self.run_one(problem_template, algorithm_template, hyper_param_dict, run_name, final_output_path, tmp_output_path, priority)
                # save problem name, algorithm name, hyperparameters and yaml files
            used_run_paths.append(final_output_path)

        experiment_fullfile = self.save_experiment(experiment_base_name, used_run_paths, start_time)
        return experiment_fullfile, used_run_paths

    def run_one(self,
                problem_template, algorithm_template,
                hyper_param_dict, run_name, final_output_path, tmp_output_path, priority=0):
        """
        inner function for run
        """
        assert(isinstance(problem_template, NamedTemplate))
        assert(isinstance(algorithm_template, NamedTemplate))
        assert(isinstance(hyper_param_dict, dict))

        contents, names = \
            self.compile_contents_and_filenames(problem_template, algorithm_template,
                                                hyper_param_dict,
                                                final_output_path, tmp_output_path)
        problem_content, algorithm_content = contents
        problem_name, algorithm_name = names
        for c in contents:
            if self._has_empty_fields(c):
                raise ValueError('Not all fields filled in content file:\n' + c)
        if not self.use_sge:
            print_named_content("Starting Experiment", "")

            if not self.DEBUG_MODE:
                raise RuntimeError("Must use SGE if not in debug mode")

        else:
            # submit to SGE
            print_named_content('Problem:', problem_content)
            print_named_content('Algorithm:', algorithm_content)

            d = {'caffe_binary_fullfile': self.caffe_binary_fullfile,
                 'caffe_root': self.caffe_root,
                 'algorithm_fullfile': algorithm_name,
                 'tmp_output_path': tmp_output_path,
                 'final_output_path': final_output_path}
            sge_script = sge_template.safe_substitute(**d)

            sge_script_name = 'run.sh'
            sge_scipt_file = os.path.join(final_output_path, sge_script_name)

            if self.DEBUG_MODE:
                print_named_content(sge_scipt_file, sge_script)
            else:
                with open(sge_scipt_file, "w") as f:
                    f.write(sge_script)

            # error_log_file = os.path.join(final_output_path, "error.log")
            # output_log_file = os.path.join(final_output_path, "output.log")
            sge_command = 'qsub -b n -V -p %d -N "%s" -wd "%s" -e "%s" -o "%s" "%s"' \
                          % (int(priority), run_name, final_output_path,
                             'error.log', 'output.log', sge_scipt_file)

            print_named_content('SGE Command:', sge_command)

            if not self.DEBUG_MODE:
                subprocess.call(sge_command, shell=True)

    def get_this_final_run_path(self, run_name):
        return os.path.join(self.data_path, run_name)

    def get_this_tmp_run_path(self, run_name):
        return os.path.join(self.tmp_output_path, run_name)

    @staticmethod
    def _has_empty_fields(yaml_str):
        return '$' in yaml_str

    def save_experiment(self, experiment_base_name, used_run_paths, start_time):
        experiment_filename = self.compile_experiment_name(experiment_base_name, start_time)
        experiment_fullfile = os.path.join(self.experiment_path, experiment_filename)

        if self.DEBUG_MODE:
            print_named_content(experiment_fullfile, '\n'.join(used_run_paths))
        else:
            with open(experiment_fullfile, 'w') as f:
                for r in used_run_paths:
                    f.write("%s\n" % r)
            return experiment_fullfile

    def read_experiment_file(self, fullfilename):
        """
        loads the channel files from the runs referred to in a given experiment file.
        """
        filename = os.path.basename(fullfilename)

        if filename == fullfilename:
            # we were only passed a filename, not a path.  load filename from the default experiment directory
            fullfilename = os.path.join(self.experiment_path, filename)

        with open(fullfilename) as f:
            data_paths = f.read().splitlines()

        experiment_name, _ = filename.split(seps['super'])

        return data_paths, experiment_name

    # def load_channels_from_experiment(self, filename):
    #     """
    #     load the channels files located in all directories in the given experiment file; return them in a Pandas Panel
    #
    #     :param filename: the name of the experiment file to load.  must be on the python path or be fully qualified
    #     :rtype: Pandas.Panel
    #     :return: a Pandas Panel in which each item as a single algorithm for the experiment file, major axis is time
    #              slice, minor axis is channel name
    #     """
    #     import pandas as pd
    #
    #     data_paths, exp_name = self.read_experiment_file(filename)
    #     data = {}
    #     n_paths = len(data_paths)
    #     for i, f in enumerate(data_paths):
    #         print 'now loading file % 4d of % 4d: %s' % (i+1, n_paths, f)
    #         df, alg_name = MonitorParser.load_channels(os.path.join(f, self.channels_filename))
    #         data[alg_name] = df
    #
    #     return pd.Panel(data), exp_name

    def compile_experiment_name(self, experiment_base_name, start_time):
        d = seps.copy()
        d['experiment_base_name'] = experiment_base_name
        d['start_time'] = start_time
        return self.experiment_filename_template.safe_substitute(d)

    def compile_run_names(self, problem_template, algorithm_template, hyper_param_dicts, start_time):
        return [self.compile_run_name(problem_template, algorithm_template, p, start_time) for p in hyper_param_dicts]

    def compile_run_name(self, problem_template, algorithm_template, hyper_param_dict, start_time):
        problem_name = problem_template.fill_name(hyper_param_dict)
        algorithm_name = algorithm_template.fill_name(hyper_param_dict)
        params = hyper_param_dict['params']
        d = seps.copy()
        d.update(locals())
        del d['self']
        if 'seed' in hyper_param_dict:
            d['seed'] = hyper_param_dict['seed']
        else:
            d['seed'] = None

        return self.run_name_template.safe_substitute(**d)

    def save_names_and_hyper_params(self,
                                    problem_template, algorithm_template,
                                    hyper_param_dict, this_run_path):
        problem_name = problem_template.fill_name(hyper_param_dict)
        algorithm_name = algorithm_template.fill_name(hyper_param_dict)
        d = hyper_param_dict.copy()
        d['problem_name'] = problem_name
        d['algorithm_name'] = algorithm_name

        hyper_param_fullfile = os.path.join(this_run_path, self.name_names_hyper_params)
        if not self.DEBUG_MODE:
            with open(hyper_param_fullfile, 'wb') as outfile:
                cPickle.dump(d, outfile, protocol=cPickle.HIGHEST_PROTOCOL)

    # def load_run_data(self, run_path, recreate_small_version=False):
    #     """ load the model and algo name file from a run directory """
    #
    #     # save small versions
    #     # try to load the small version, if that's not possible, or we've been ordered to recreate the small version
    #     # no matter what,
    #
    #     full_model_file = os.path.join(run_path, self.name_trained_model)
    #     small_model_file = os.path.join(run_path, self.trained_model_small)
    #     if os.path.isfile(small_model_file) and not recreate_small_version:
    #         model = self.load_pkl(small_model_file)
    #     else:
    #         model = self.load_pkl(full_model_file)
    #         self.shrink_model(model)
    #         self.save_pkl(small_model_file, model)
    #     names = self.load_pkl(os.path.join(run_path, self.name_names_hyper_params))
    #     return model, names

    # @staticmethod
    # def load_pkl(filename):
    #     with open(filename, 'rb') as f:
    #         model =  cPickle.load(f)
    #     return model
    #
    # @staticmethod
    # def save_pkl(filename, object):
    #     with open(filename, 'wb') as outfile:
    #         cPickle.dump(object, outfile, protocol=cPickle.HIGHEST_PROTOCOL)

    def save_problem_and_algorithm(self, problem_template, algorithm_template, hyper_param_dict, final_output_path, tmp_output_path):
        contents, save_filenames = \
            self.compile_contents_and_filenames(problem_template, algorithm_template,
                                                hyper_param_dict, final_output_path, tmp_output_path)

        for content, filename in zip(contents, save_filenames):
            if self.DEBUG_MODE:
                print_named_content(filename, content)
            else:
                with open(filename, 'w') as outfile:
                    outfile.write(content)

    def compile_contents_and_filenames(self,
                                       problem_template, algorithm_template,
                                       hyper_param_dict, final_output_path, tmp_output_path):
        # caffe only needs the algorithm content to know the problem file's path but including
        # both here for both
        d = hyper_param_dict.copy()
        d['problem_fullfile'] = os.path.join(final_output_path, self.name_problem_file)
        d['algorithm_fullfile'] = os.path.join(final_output_path, self.name_algorithm_file)

        d['final_output_path'] = final_output_path
        d['tmp_output_path'] = tmp_output_path

        d['caffe_root'] = self.caffe_root
        d['caffe_binary_fullfile'] = self.caffe_binary_fullfile

        contents = [problem_template.fill_content(d)]
        contents += [algorithm_template.fill_content(d)]

        names = [d['problem_fullfile'], d['algorithm_fullfile']]

        return contents, names

    @staticmethod
    def glob_escape(in_str):
        """ escape a string so that it can be used in glob.glob """
        special_chars = ['?', '*', '[']
        for c in special_chars:
            in_str = in_str.replace(c, '[%s]' % c)
        return in_str

    @classmethod
    def compatible_name(cls, run_name):
        """
        return the base of this run_name for finding compatible runs.
        the base is the run_name without the random seed and date/time info

        2014.11.24:
        changing this to care about seed too.  date/time has already been eliminated
        so this means the only compatible run name will be your exact run name
        """
        # compatible_name, _ = run_name.split(seps['super'])
        return run_name

    def find_compatible_run(self, run_name):
        """
        2014.11.24:
        changing this to look for directories with both trained model files in them since on SGE i now create the
        directories before running all jobs but the trained model only gets copied back once the entire run completes
        successfully
        still returns the run path without self.name_trained_model appended as it did before
        """
        compatible_name = self.compatible_name(run_name)
        search_str = self.glob_escape(os.path.join(self.data_path, compatible_name, self.name_trained_model))
        # yes, this should be append not extend.  we're returning a list of lists
        compatible_trained_models = glob.glob(search_str)
        compatible_runs = [os.path.dirname(f) for f in compatible_trained_models]
        return compatible_runs

    def find_compatible_runs(self, run_names):
        return [self.find_compatible_run(n) for n in run_names]

    def offer_compatible_runs(self, run_names, all_compatible_runs):
        print ' '
        print 'Compatible runs:'
        default_choices = self._print_compatible_choices(run_names, all_compatible_runs)
        if all(c == 0 for c in default_choices):
            print 'No compatible runs located'
            choices = default_choices
        else:
            print 'Compatible runs located for one or more runs in this experiment.'
            print 'Do you want to use any of them in place of creating a new run?'
            print 'Enter a list of with your choice for each run.  An entry of 0 in the list means to make a new run'
            print 'for that setting rather than to use an existing one.  Instead of a list, you can also just input'
            print 'the single number 0 to use none of the existing runs and create all new ones instead.'
            print ' '
            try:
                choices = input('CHOICES, default=%s: ' % default_choices)
            except SyntaxError:
                choices = default_choices
            if choices == 0:
                choices = [0] * len(run_names)
        print 'Accepting choices:', choices
        return choices

    def _print_compatible_choices(self, run_names, all_compatible_runs):
        default_choices = [0] * len(run_names)
        for i, (r, crs) in enumerate(zip(run_names, all_compatible_runs)):
            compatible_name = self.compatible_name(r)
            print i, compatible_name
            default_choices[i] = len(crs)  # choose the longest running by default
            for j, c in enumerate(crs):
                print '  ', j+1, c
            print ' '
        return default_choices

    @staticmethod
    def shrink_model(model):
        """
        delete the large attributes of a pylearn2 model in place.
        the new version is quicker to load/save and still has all the information i want for plotting
        """
        del model.layers

class NamedTemplate(object):
    """ named template """
    def __init__(self, name_template, content_template):
        self._name_template = self._validate_template(name_template)
        self._content_template = self._validate_template(content_template)

    @staticmethod
    def _validate_template(template):
        if isinstance(template, Template):
            return template
        elif isinstance(template, str):
            return Template(template)
        else:
            raise ValueError('template must be either a str or a Template object')

    def fill(self, hyper_params):
        return self.fill_name(hyper_params), self.fill_content(hyper_params)

    def fill_name(self, hyper_params):
        return self._name_template.safe_substitute(hyper_params)

    def fill_content(self, hyper_params):
        content = self._content_template.safe_substitute(hyper_params)
        # if self.has_unfilled_fields(content):
        #     content = Template(content)
        return content

    @property
    def content_template(self):
        return self._content_template.template

    @property
    def name_template(self):
        return self._name_template.template

    # @staticmethod
    # def has_unfilled_fields(string):
    #     return '${' in string


# helper functions for constructing settings dictionaries
def cross_dict(params_to_cross_dict):
    """
    take the cross product of the parameters in params_to_cross_dict.

    :param dict params_to_cross_dict: each key is a parameter name, each value is an iterable of parameter values
    :return: a list of dictionaries, each of which one set of of k:v pairs from the cross product of all options
             in params_to_cross_dict
    :rtype: [dict]
    """
    import itertools
    ks = params_to_cross_dict.keys()
    params_to_cross = params_to_cross_dict.values()
    param_sets = itertools.product(*params_to_cross)

    out_dicts = []
    for param_set in param_sets:
        o = {k: v for k, v in zip(ks, param_set)}
        out_dicts.append(o)

    return out_dicts

def append_dicts(base_dict, dicts_to_append):
    """
    append update base_dict with each dict in dicts_to_append
    :param dict base_dict: the starting dictionary to be updated
    :param list(dict) dicts_to_append: n dicts, each of which will update base_dict in turn
    :return: list of n dicts, each of which is base_dict updated with one dict from dicts_to_append
    :rtype: list(dict)
    """
    if isinstance(dicts_to_append, dict):
        dicts_to_append = (dicts_to_append,)

    out_dicts = []
    for d in dicts_to_append:
        o = base_dict.copy()
        o.update(d)
        out_dicts.append(o)

    return out_dicts

problem_content_template = Template("""
name: "MNIST_500-300_sigmoid_softmax"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.0039215684
  }
  data_param {
    source: "/storage/code/caffe/examples/mnist/mnist_train_lmdb"
    batch_size: ${train_batch_size}
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
    stage: "test-on-train"
  }
  transform_param {
    scale: 0.0039215684
  }
  data_param {
    source: "/storage/code/caffe/examples/mnist/mnist_train_lmdb"
    batch_size: ${train_batch_size}
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
    stage: "test-on-test"
  }
  transform_param {
    scale: 0.0039215684
  }
  data_param {
    source: "/storage/code/caffe/examples/mnist/mnist_test_lmdb"
    batch_size: ${test_batch_size}
    backend: LMDB
  }
}
layer {
  name: "h0_ip"
  type: "InnerProduct"
  bottom: "data"
  top: "h0_ip"
  param {
    lr_mult: 1
    decay_mult: 0
  }
  param {
    lr_mult: 1
    decay_mult: 0
  }
  inner_product_param {
    num_output: ${n_neurons_h0}
    weight_filler {
      type: "gaussian"
      std: .01
      sparse: 50
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "h0"
  type: "Sigmoid"
  bottom: "h0_ip"
  top: "h0"
}
layer {
  name: "h1_ip"
  type: "InnerProduct"
  bottom: "h0"
  top: "h1_ip"
  param {
    lr_mult: 1
    decay_mult: 0
  }
  param {
    lr_mult: 1
    decay_mult: 0
  }
  inner_product_param {
    num_output: ${n_neurons_h1}
    weight_filler {
      type: "gaussian"
      std: .01
      sparse: 30
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "h1"
  type: "Sigmoid"
  bottom: "h1_ip"
  top: "h1"
}
layer {
  name: "y_ip"
  type: "InnerProduct"
  bottom: "h1"
  top: "y_ip"
  param {
    lr_mult: 1
    decay_mult: 0
  }
  param {
    lr_mult: 1
    decay_mult: 0
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "gaussian"
      std: .01
      sparse: 0
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "y_ip"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "y_ip"
  bottom: "label"
  top: "loss"
}
""")

algorithm_content_template = Template("""
net: "${problem_fullfile}"

test_state: { stage: 'test-on-train' }
test_iter: ${n_test_on_train_iters}
test_state: { stage: 'test-on-test' }
test_iter: ${n_test_on_test_iters}

test_interval: ${n_iters_before_test}
test_compute_loss: true

base_lr: ${alpha}
lr_policy: "fixed"
momentum: ${momentum}
#gamma: 0.1
#stepsize: 10000

display: ${n_iters_before_display}
max_iter: ${n_max_iters}
snapshot: ${n_iters_before_snapshot}
snapshot_prefix: "${tmp_output_path}/snapshot"
# solver mode: CPU or GPU
solver_mode: GPU
solver_type: SGD

random_seed: ${seed}
""")

def _test_run_offer():
    """
    putting this here because it requires human intervention so can't go in tests

    you can run it and make sure that it uses the runs you tell it to and makes new runs when it's supposed to.
    the run choice prompt describes what it's suppsed to do pretty well

    if it doesn't find any compatible runs, though, it will just skip that screen.  you can just run the same thing
    twice to see it in action.
    """
    base_params = {'n_train': 10000,
                   'n_h0': 12,
                   'save_freq': 1,
                   'batch_size': 100,
                   'init_momentum': .5,
                   'n_epochs': 2,
    }
    d = { 'learning_rate': [0.1, 0.01], 'seed': [4, 5] }
    ds = cross_dict(d)
    hyper_param_dicts = append_dicts(base_params, ds)

    experiment_name = 'TEST_EXPERIMENT'
    e = Experiment()
    problem_template = NamedTemplate(problem_name_template_str, problem_content_template_str)
    algorithm_template = NamedTemplate(algorithm_name_template_str, algorithm_content_template_str)

    experiment_fullfile, used_paths = \
        e.run(experiment_name, problem_template, algorithm_template, hyper_param_dicts, offer_compatible_runs=True)

    # ensure that correct files were created
    assert os.path.exists(experiment_fullfile)
    expected_files = [e.name_algorithm_file,
                      e.name_problem_file,
                      e.name_names_hyper_params,
                      e.name_trained_model,
                      ]
    for run_path in used_paths:
        assert os.path.exists(run_path)
        for f in expected_files:
            assert os.path.exists(os.path.join(run_path, f))

# class TemplatedFile(object):
#     def __init__(self, output_path_template, filename_template, content_template):
#         self.output_path_template = self._validate_template(output_path_template)
#         self.filename_template = self._validate_template(filename_template)
#         self.content_template = self._validate_template(content_template)
#
#     @staticmethod
#     def _validate_template(template):
#         if isinstance(template, basestring):
#             template = Template(template)
#         assert(isinstance(template, Template))
#         return template
#
#     def fullfile(self, hyper_param_dict):
#         output_path = self.output_path_template.safe_substitute(hyper_param_dict)
#         filename = self.filename_template.safe_substitute(hyper_param_dict)
#
#         return os.path.join(output_path, filename)
#
#     def save(self, hyper_param_dict):
#         output_fullfile = self.fullfile(hyper_param_dict)
#         content = self.content_template.safe_substitute(hyper_param_dict)
#
#         with open(output_fullfile, "w") as f:
#             f.write(content)
#
#         return output_fullfile, content


if __name__ == '__main__':
    batch_size = 100
    n_data_train = 60000
    n_data_test = 10000

    n_iter_train = n_data_train / batch_size
    n_epochs = 500

    experiment_base_name = 'TEST-EXPERIMENT'

    base_params = {
        'train_batch_size': batch_size,
        'test_batch_size': batch_size,
        'n_test_on_train_iters': n_iter_train,
        'n_test_on_test_iters': n_data_test / batch_size,
        'n_iters_before_test': n_iter_train,
        'n_neurons_h0': 500,
        'n_neurons_h1': 300,
        'n_iters_before_display': 100,
        'n_max_iters': n_iter_train * n_epochs,
        'n_iters_before_snapshot': n_iter_train,
        'params': experiment_base_name,
    }
    d = {'alpha': [0.1, 0.01],
         'momentum': [.5],
         'seed': [9],
         }
    ds = cross_dict(d)
    hyper_param_dicts = append_dicts(base_params, ds)

    problem_content = problem_content_template
    algorithm_content = algorithm_content_template
    problem_name = Template('MNIST_MPL(${n_neurons_h0}, ${n_neurons_h1})')
    algorithm_name = Template('SGD(a=${alpha}, m=${momentum})')

    problem_template = NamedTemplate(problem_name, problem_content)
    algorithm_template = NamedTemplate(algorithm_name, algorithm_content)

    e = Experiment(use_sge=False, DEBUG_MODE=True)
    e.run(experiment_base_name, problem_template, algorithm_template, hyper_param_dicts,
          offer_compatible_runs=False)

    # _test_run_offer()

