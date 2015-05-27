from __future__ import division
from math import floor
import os
import sys
import itertools

import numpy as np
import pylab

this_path = os.path.dirname(os.path.realpath(__file__))
sys.path += [this_path]

from experiment import Experiment, seps

def data_dir_2_alg_name(*args):
    if len(args) == 2:
        # first arg is 'self'
        data_dir_name = args[1]
    else:
        data_dir_name = args[0]
    # this is a hack to get the right separator
    alg_name, _ = data_dir_name.split(seps['super'])
    _, alg_name = alg_name.split(seps['major'])
    return alg_name

def read_experiment_file(fullfilename):
    """
    loads the channel files from the runs referred to in a given experiment file.
    """
    filename = os.path.basename(fullfilename)

    with open(fullfilename) as f:
        data_paths = f.read().splitlines()

    experiment_name, _ = filename.split(seps['super'])

    return data_paths, experiment_name

def load_parsed_df(channels_filename):
    """
    load a channels_filename data frame saved by save_channels

    :param channels_filename: the name of the channels_filename file to load
    :rtype: pandas.DataFrame
    :return: DataFrame representing all the channels_filename in a model, time slice is the index and channel names are
             the columns and the name of the algorithm that generated this run, i.e. the name of the directory in which
             the channels file is contained
    """
    import pandas as pd
    if hasattr(pd, 'read_pickle'):
        # true for recent pandas
        df = pd.read_pickle(channels_filename)
    else:
        # true for old pandas
        df = pd.load(channels_filename)
    algorithm_name = os.path.basename(os.path.dirname(channels_filename))
    return df, algorithm_name

def replace_storage_with_this_path(fullfile):
    if fullfile.startswith('/storage'):
        output_path = fullfile.split('/', 2)[2]
        return os.path.join(this_path, output_path)
    else:
        return fullfile

def load_parsed_dfs_from_experiment(exp_filename, df_filename='parsed_log_df.pkl'):
    """
    load the parsed log dfs located in all directories in the given experiment file; return them in a Pandas Panel

    :param exp_filename: the name of the experiment file to load.  must be on the python path or be fully qualified
    :rtype: Pandas.Panel
    :return: a Pandas Panel in which each item as a single algorithm for the experiment file, major axis is time
             slice, minor axis is channel name
    """
    import pandas as pd

    data_paths, exp_name = read_experiment_file(exp_filename)
    data = {}
    n_paths = len(data_paths)
    for i, f in enumerate(data_paths):
        f = replace_storage_with_this_path(f)
        print 'now loading file % 4d of % 4d: %s' % (i+1, n_paths, f)
        df, alg_name = load_parsed_df(os.path.join(f, df_filename))
        data[alg_name] = df

    return pd.Panel(data), exp_name


class Plotter(object):
    x_names_2_labels = {
        'batches_seen': 'Batches Seen',
        'seconds_seen': 'Seconds',
        # 'minutes_seen': 'Minutes',
        # 'hours_seen': 'Hours',
        'epochs_seen': 'Epochs',
        'datapoints_seen': 'Datapoints Seen',
    }

    # this translates from <plot code variable name>: <plot label name>, <take semilogy>
    # override this dict to change plot labels or semilogs
    y_names_2_labels = {
        'test_y_misclass': ('Test Error Rate', False) ,
        'valid_y_misclass': ('Validation Error Rate', False),
        'train_y_misclass': ('Training Error Rate', False),

        'test_objective': ('Test Objective', True),
        'valid_objective': ('Validation Objective', True),
        'train_objective': ('Training Objective', True),
        'test_y_nll': ('Test Objective', True),
        'valid_y_nll': ('Validation Objective', True),
        'train_y_nll': ('Training Objective', True),

        'learning_rate': ('Learning Rate', True),
        'grad_norm': ('Gradient Norm', True),
        'grad_norm_current': ('Gradient Norm\n(Most Recent)', True),

        'momentum': ('Momentum', False),
    }

    default_colors = ['k', 'r', 'g', 'b', 'c', 'm', 'y']
    default_styles = ['-', '--', ':',  '-.'] * 4
    styles = [sc[0] for sc in itertools.product(default_styles, default_colors)]
    colors = [sc[1] for sc in itertools.product(default_styles, default_colors)]

    default_plots_path_addon = 'plots'

    def __init__(self, output_path=None, linewidth=1.5):
        e = Experiment(output_path)
        self.linewidth = linewidth
        if output_path is None:
            self.output_path = os.path.join(e.final_output_path, self.default_plots_path_addon)
        else:
            self.output_path = output_path
        e.makedir(self.output_path)


class ScatterPlotter(Plotter):
    default_vars_to_plot = ('test_y_misclass', 'train_y_nll',)

    def __init__(self, output_path=None, linewidth=1.5):
        super(ScatterPlotter, self).__init__(output_path, linewidth)

    @staticmethod
    def default_marker_map(algorithm_name):
        """ given an algorithm name, return (<marker style>, <marker color>, <marker size>, <zorder>, <label for legend>)"""
        # DEFAULT_SIZE = 100

        import re
        p = re.compile(r".*batch=(?P<batchsize>\d+).*")
        m = p.match(algorithm_name)
        batch_size = int(m.group('batchsize'))

        display_size = batch_size * .5 + 16

        # higher zorder means display further to the front
        an = algorithm_name.lower()
        if 'ducb' in an:
            # marker style, marker color, marker size, zorder, label for legend
            return 'o', 'r', display_size, 10, 'DUCB, batch =% 5d' % batch_size
        elif 'adadelta' in an:
            return 'o', 'g', display_size, 5, 'AdaDelta, batch =% 5d' % batch_size
        elif 'sgd' in an:
            # scale for perceptual size
            return 'o', 'b', display_size, 1, 'SGD, batch =% 5d' % batch_size
        else:
            raise ValueError("Algoritm name must contain one of 'ducb', 'adadelta', or 'sgd'.  " +
                             "You gave: %s" % algorithm_name)

    @staticmethod
    def default_error_bar_fn(data):
        """
        takes an iterable representing one dimension of data (i.e. x-data or y-data) from a single datapoint in a
        grouped scatter plot.  so data should be e.g. all the test errors from a single algorithm across all random seeds.

        returns the middle, low and high values for the point and error bars for that group of data.
        """
        if len(data) == 1:
            return data[0], 0, 0
        middle = np.median(data)
        low = np.min(data)
        high = np.max(data)
        return middle, np.abs(low-middle), np.abs(high-middle)

    @staticmethod
    def group_indices(data, keyfunc):
        """
        group the objects in data into groups based on sort key.

        inputs:
            data: an iterable of the objects to sort
            keyfunc: callable that takes an object in data and returns the groupby key for that object

        returns:
            (indices, unique_keys) where
            unique_keys: a list of the unique keys in data and
            indices: a list in which each element is a list of the indices of the objects in 'data' which had the groupby
                     key from the corresponding entry in unique_keys
            indices and unique_keys will be the same length

        e.g.:
            data =  ('asdf', 'fff', 'asdd', 'aqer', 'asdf', 'f', '234f', '23rq', 'asdfa')
            keyfunc = lambda x: x[-1] # the sort key for each element in 'data' is its last character
            indices, unique_keys = group_indices(data, keyfunc)
            print unique_keys
            # ['a', 'd', 'f', 'q', 'r']
            print indices
            # [[8], [2], [0, 1, 4, 5, 6], [7], [3]]
        """
        from itertools import groupby
        indices = []
        unique_keys = []
        # wrap keyfunc so it can ignore the enumerate index
        wrappedkeyfunc = lambda x: keyfunc(x[1])
        ind_objs = sorted(enumerate(data), key=wrappedkeyfunc)
        for k, g in groupby(ind_objs, wrappedkeyfunc):
            # k is the value of the sort key returned by keyfunc
            # g is an iterable in which each element is a tuple of the form (<original sort index>, <object>)
            # now we're going to strip away the objects and just keep a list of the original sort indices
            indices.append([i for i, o in g])
            unique_keys.append(k)
        return indices, unique_keys

    default_groupby_key = data_dir_2_alg_name

    def plot_from_experiment(self,
                             experiment_filename,
                             plot_base_name=None,
                             plot_addon_name=None,
                             vars_to_plot=None,
                             time_slices=(-1,),
                             marker_map_fn=None,
                             do_group=False,
                             groupby_key=None,
                             error_bar_fn=None,
                             sync_margins_across_times=True,
                             x_lims = (None, None),
                             y_lims = (None, None),
                             **kwargs):
        panel, exp_name = load_parsed_dfs_from_experiment(experiment_filename)
        if plot_base_name is None:
            plot_base_name = exp_name
        if plot_addon_name is not None:
            plot_base_name = plot_base_name + plot_addon_name
        return self.plot(panel=panel,
                         plot_base_name=plot_base_name,
                         vars_to_plot=vars_to_plot,
                         time_slices=time_slices,
                         marker_map_fn=marker_map_fn,
                         do_group=do_group,
                         groupby_key=groupby_key,
                         error_bar_fn=error_bar_fn,
                         sync_margins_across_times=sync_margins_across_times,
                         x_lims = x_lims,
                         y_lims = y_lims,
                         **kwargs)

    def plot(self,
             panel,
             plot_base_name,
             vars_to_plot=None,
             time_slices=(-1,),
             save_formats=('png',),
             fig_inches=(12, 9),
             marker_map_fn=None,
             do_group=False,
             groupby_key=None,
             error_bar_fn=None,
             sync_margins_across_times=True,
             x_lims = (None, None),
             y_lims = (None, None),
             ):

        if marker_map_fn is None:
            marker_map_fn = self.default_marker_map

        if do_group:
            if groupby_key is None:
                groupby_key = self.default_groupby_key
            if error_bar_fn is None:
                error_bar_fn = self.default_error_bar_fn

        if vars_to_plot is None:
            vars_to_plot = self.default_vars_to_plot

        if len(vars_to_plot) != 2:
            raise ValueError('vars_to_plot must be a tuple of two variable names')

        # # get data
        all_marker_styles = [marker_map_fn(k) for k in panel.items]
        # this is not the cleanest way of doing this but i'm making the minimal change here to get plotter working
        # with dataframes since they load so much faster than the full model or even small model files i was using
        # before.  would be better to go back and rewrite all Plotters to work with channel DataFrames from the start
        all_xvals = panel.minor_xs(vars_to_plot[0]).as_matrix().T
        all_yvals = panel.minor_xs(vars_to_plot[1]).as_matrix().T

        axis_labels = [self.y_names_2_labels[var][0] for var in vars_to_plot]
        semilogs = [self.y_names_2_labels[var][1] for var in vars_to_plot]

        SMALL_VAL = 1e-12
        if semilogs[0]:
            all_xvals += SMALL_VAL
        if semilogs[1]:
            all_yvals += SMALL_VAL

        all_figs = []
        all_axes = []

        # min / max  x / y across all plots.  used to sync margins of multiple time slices
        grand_min_x = np.inf
        grand_min_y = np.inf
        grand_max_x = -np.inf
        grand_max_y = -np.inf

        all_full_plot_names = []

        # for each plot (each time slice gets its own plot)
        for time_slice in time_slices:
            print 'Plotting time_slice % 3d' % time_slice
            fig, axis = pylab.subplots(1, sharex=True, sharey=True)
            all_figs.append(fig)
            all_axes.append(axis)
            fig.hold(True)

            if do_group:
                all_points, marker_labels, minx, miny, maxx, maxy, = \
                    self._grouped_scatter_plot(all_marker_styles,
                                               all_xvals,
                                               all_yvals,
                                               axis,
                                               time_slice,
                                               panel.items,
                                               groupby_key,
                                               error_bar_fn)
            else:
                all_points, marker_labels, minx, miny, maxx, maxy = \
                    self._ungrouped_scatter_plot(all_marker_styles,
                                                 all_xvals,
                                                 all_yvals,
                                                 axis,
                                                 time_slice)

            self.scale_xy_axes(axis, minx, miny, maxx, maxy, semilogs)

            # min/max x/y across all plots
            grand_min_x = min(grand_min_x, minx)
            grand_min_y = min(grand_min_y, miny)
            grand_max_x = max(grand_max_x, maxx)
            grand_max_y = max(grand_max_y, maxy)

            # axis labels
            axis.set_xlabel(axis_labels[0])
            axis.set_ylabel(axis_labels[1])

            # window title and size
            plot_name = '%s by %s, epoch % 5d' % (axis_labels[0], axis_labels[1], time_slice)
            full_plot_name = '%s -- %s' % (plot_base_name, plot_name)
            fig.canvas.set_window_title(full_plot_name)
            fig.set_size_inches(*fig_inches)
            fig.suptitle(plot_name)

            all_full_plot_names.append(full_plot_name)

            # sort both marker_labels and all_points by marker_labels
            marker_labels, all_points = zip(*sorted(zip(marker_labels, all_points), key=lambda t: t[0]))

            # legend
            fig.legend(all_points,
                       marker_labels,
                       scatterpoints=1,
                       loc='upper right',
                       )

            # set log scales
            if semilogs[0]:
                axis.semilogx()
            if semilogs[1]:
                axis.semilogy()

        out_images = []
        for fig, axis, full_plot_name in zip(all_figs, all_axes, all_full_plot_names):
            if sync_margins_across_times:
                self.scale_xy_axes(axis, grand_min_x, grand_min_y, grand_max_x, grand_max_y, semilogs)
            # if you've passed in an axis scaling, it takes presedence over the automatic version
            use_x_lims = list(axis.get_xlim())
            use_y_lims = list(axis.get_ylim())
            if x_lims[0] is not None: use_x_lims[0] = x_lims[0]
            if x_lims[1] is not None: use_x_lims[1] = x_lims[1]
            if y_lims[0] is not None: use_y_lims[0] = y_lims[0]
            if y_lims[1] is not None: use_y_lims[1] = y_lims[1]
            axis.set_xlim(x_lims)
            axis.set_ylim(y_lims)

            # save
            if save_formats is not None:
                for fmt in save_formats:
                    outpath = os.path.join(self.output_path, full_plot_name+'.'+fmt)
                    out_images.append(outpath)
                    print 'Saving', outpath
                    fig.savefig(outpath, bbox_inches='tight', dpi=300)

        print 'ScatterPlotter: plots are up'
        # pylab.show()
        return out_images


    def scale_xy_axes(self, axis, minx, miny, maxx, maxy, semilogs, buffer_mx = 0.1):
        # figure out axis ranges.  put a default 10% buffer on each side
        # if axis is on a log scale, then scaling is multiplicative, otherwise additive
        if semilogs[0]:
            rangex = np.log10(maxx / minx)
            bufferx = buffer_mx * rangex
            use_min_x = minx / pow(10, bufferx)
            use_max_x = maxx * pow(10, bufferx)
        else:
            rangex = maxx - minx
            bufferx = buffer_mx * rangex
            use_min_x = minx - bufferx
            use_max_x = maxx + bufferx
        if semilogs[1]:
            rangey = np.log10(maxy / miny)
            buffery = buffer_mx * rangey
            use_min_y = miny / pow(10, buffery)
            use_max_y = maxy * pow(10, buffery)
        else:
            rangey = maxy - miny
            buffery = buffer_mx * rangey
            use_min_y = miny - buffery
            use_max_y = maxy + buffery
        axis.set_xlim([use_min_x, use_max_x])
        axis.set_ylim([use_min_y, use_max_y])

    def _ungrouped_scatter_plot(self, all_marker_styles, all_xvals, all_yvals, axis, time_slice):
        # the meat of an ungrouped scatter plot.   points are addeed to the scatter plot one style at a time
        # returns all_points and marker_labels for later use by the legend code
        all_points = []
        unique_marker_styles = set(all_marker_styles)
        marker_labels = [label for (marker, color, size, zorder, label) in unique_marker_styles]
        # display points
        for marker_style in unique_marker_styles:
            row_inds = np.array([marker_style == ms for ms in all_marker_styles])
            xvals = all_xvals[row_inds, time_slice]
            yvals = all_yvals[row_inds, time_slice]
            marker, color, size, zorder, label = marker_style
            ALPHA = .6
            LINE_WIDTHS = 0
            all_points.append(axis.scatter(xvals, yvals, s=size, c=color, marker=marker, zorder=zorder,
                                           linewidths=LINE_WIDTHS, alpha=ALPHA))
        minx = np.min(all_xvals[:, time_slice])
        miny = np.min(all_yvals[:, time_slice])
        maxx = np.max(all_xvals[:, time_slice])
        maxy = np.max(all_yvals[:, time_slice])
        return all_points, marker_labels, minx, miny, maxx, maxy

    def _grouped_scatter_plot(self, all_marker_styles, all_xvals, all_yvals, axis, time_slice, alg_names, groupby_key, error_bar_fn):
        # the meat of a grouped scatter plot.
        #
        # the idea is that all runs which come from the same method but have different random seeds should all
        # be considered together.  so we group by model.algorithm_name (which is cheat-set by the plot_from_experiment
        # method) since model.algorithm_name should be a complete description of the algorithm and doesn't include any
        # reference to random seed.
        #
        # there are two levels of grouping at play here.
        # first: unique_marker_styles which is for display / legend
        # second: by the callable gropuby_key which takes a model and returns a key to group by.

        indices, unique_keys = self.group_indices(alg_names, groupby_key)

        all_points = []
        marker_labels = []
        minx = miny = np.inf
        maxx = maxy = -np.inf
        for key, inds in zip(unique_keys, indices):

            # here, i'm assuming that all_marker_styles is the same for every element in the group,
            # i.e. each run of a given algorithm across all its random seeds.  this is enforced by writing a
            # marker_map function that is compatible with your groupby_key function
            marker, color, size, zorder, label = all_marker_styles[inds[0]]

            xdata = all_xvals[np.array([inds]), time_slice].flatten()
            ydata = all_yvals[np.array([inds]), time_slice].flatten()
            xmid, xlow, xhigh = error_bar_fn(xdata)
            ymid, ylow, yhigh = error_bar_fn(ydata)
            xerrs = np.array([[xlow], [xhigh]])
            yerrs = np.array([[ylow], [yhigh]])

            minx = min(xmid-xlow, minx)
            miny = min(ymid-ylow, miny)
            maxx = max(xmid+xhigh, maxx)
            maxy = max(ymid+yhigh, maxy)

            # print 'ScatterPlotter group size -- x: %f [%f, %f], y: %f [%f, %f]' % (xmid, xmid-xlow, xmid+xhigh, ymid, ymid-ylow, ymid+yhigh)

            # taking sqrt because scatter does size by area, errorbar does by radius
            MARKER_EDGE_WEIGHT = 1
            ALPHA = .6
            h = axis.errorbar(x=xmid, y=ymid, yerr=yerrs, xerr=xerrs, zorder=zorder,
                              marker=marker, mfc=color, ms=np.sqrt(size), mew=MARKER_EDGE_WEIGHT,
                              ecolor='k', elinewidth=1.5, alpha=ALPHA)
            if label not in marker_labels:
                all_points.append(h)
                marker_labels.append(label)
        return all_points, marker_labels, minx, miny, maxx, maxy


class TimeSeriesPlotter(Plotter):
    def __init__(self, ):
        super(TimeSeriesPlotter, self).__init__()


    @staticmethod
    def default_line_style_map(algorithm_name):
        """ given an algorithm name, return (<color>, <linestyle>, <linewidth>, <alpha>, <zorder>, <line label>)"""

        import re
        p = re.compile(r".*batch=(?P<batchsize>\d+).*")
        m = p.match(algorithm_name)
        batch_size = int(m.group('batchsize'))

        p = re.compile(r".*n=(?P<n_alphas>\d+).*")
        m = p.match(algorithm_name)
        if m:
            n_alphas = int(m.group('n_alphas'))
        else:
            n_alphas = None

        MIN_BATCH_SIZE = 50
        MAX_BATCH_SIZE = 500
        # MIN_BATCH_SIZE = 500
        # MAX_BATCH_SIZE = 4000
        minlog = np.log2(MIN_BATCH_SIZE)
        maxlog = np.log2(MAX_BATCH_SIZE)

        # line_width = 2
        # line_width = (np.log2(batch_size) - minlog) / 2 + 1
        line_width = 2
        # step 1: put cmap_index between 0 and 1
        # if n_alphas is not None:
        #     cmap_index = n_alphas / 400 + .2
        # else:
        #     cmap_index = (np.log2(batch_size) - minlog ) / (maxlog - minlog)
        cmap_index = (np.log2(batch_size) - minlog ) / (maxlog - minlog)

        alpha = .9
        style='-'

        # see http://matplotlib.org/examples/color/colormaps_reference.html for colors

        # higher zorder means display further to the front
        an = algorithm_name.lower()
        if 'ducb' in an:
            # step 2: remap values to the range i want for this map
            MIN_CMAP = .33
            MAX_CMAP = .9
            cmap_index = cmap_index * (MAX_CMAP - MIN_CMAP) + MIN_CMAP

            # <color>, <linestyle>, <linewidth>, <alpha>, <zorder>, <line label>
            cm = pylab.get_cmap('afmhot_r')
            print 'ducb: ', an
            return cm(cmap_index), style, line_width, alpha, 10, 'DUCB, batch =% 5d' % batch_size
        elif 'lc' in an:
            # step 2: remap values to the range i want for this map
            MIN_CMAP = .33
            MAX_CMAP = .9
            cmap_index = cmap_index * (MAX_CMAP - MIN_CMAP) + MIN_CMAP

            # <color>, <linestyle>, <linewidth>, <alpha>, <zorder>, <line label>
            cm = pylab.get_cmap('afmhot_r')
            print 'lc: ', an
            return cm(cmap_index), style, line_width, alpha, 10, 'LineCurrent+1, batch =% 5d, n=% 3d' % (batch_size, n_alphas )
        elif 'adl' in an:
            MIN_CMAP = .33
            MAX_CMAP = .9
            cmap_index = cmap_index * (MAX_CMAP - MIN_CMAP) + MIN_CMAP
            cm = pylab.get_cmap('afmhot_r')
            print 'adl: ', an
            return cm(cmap_index), style, line_width, alpha, 5, 'ADL, batch =% 5d, n=% 3d' % (batch_size, n_alphas )
        elif 'adadelta' in an or 'ad' in an:
            MIN_CMAP = .5
            MAX_CMAP = .9
            cmap_index = cmap_index * (MAX_CMAP - MIN_CMAP) + MIN_CMAP
            cm = pylab.get_cmap('BuGn')
            print 'adadelta: ', an
            return cm(cmap_index), style, line_width, alpha, 5, 'AdaDelta, batch =% 5d' % batch_size
        elif 'agl' in an:
            # step 2: remap values to the range i want for this map
            MIN_CMAP = .33
            MAX_CMAP = .8
            cmap_index = cmap_index * (MAX_CMAP - MIN_CMAP) + MIN_CMAP

            # <color>, <linestyle>, <linewidth>, <alpha>, <zorder>, <line label>
            cm = pylab.get_cmap('PuRd')
            # cm = pylab.get_cmap('afmhot_r')
            print 'agl: ', an
            return cm(cmap_index), style, line_width, alpha, 10, 'AGL, batch =% 5d, n=% 3d' % (batch_size, n_alphas )
        elif 'ag' in an:
            MIN_CMAP = .7
            MAX_CMAP = 1.3
            cmap_index = cmap_index * (MAX_CMAP - MIN_CMAP) + MIN_CMAP
            cm = pylab.get_cmap('BuPu')
            print 'ag: ', an
            return cm(cmap_index), style, line_width, alpha, 1, 'AdaGrad, batch =% 5d' % batch_size
        elif 'sgd' in an:
            MIN_CMAP = .7
            MAX_CMAP = 1.3
            cmap_index = cmap_index * (MAX_CMAP - MIN_CMAP) + MIN_CMAP
            cm = pylab.get_cmap('Blues')
            print 'sgd: ', an
            return cm(cmap_index), style, line_width, alpha, 1, 'SGD, batch =% 5d' % batch_size
        else:
            raise ValueError("Algoritm name must contain one of 'ducb', 'adadelta', or 'sgd'.  " +
                             "You gave: %s" % algorithm_name)


    default_vars_to_plot = (
        ('epochs_seen', ['test_y_misclass', 'train_objective', 'learning_rate', 'grad_norm']),
        ('seconds_seen', ['test_y_misclass', 'train_objective', 'learning_rate', 'grad_norm']),
    )

    default_ylims = [[0, .10], None, None, None]

    legend_width_mx = 0.70

    def plot_from_experiment(self, experiment_filename, plot_base_name=None, vars_to_plot=None, **kwargs):
        panel, exp_name = load_parsed_dfs_from_experiment(experiment_filename)
        if plot_base_name is None:
            plot_base_name = exp_name

        return self.plot(panel=panel,
                         vars_to_plot=vars_to_plot,
                         plot_base_name=plot_base_name,
                         **kwargs)

    def plot(self,
             panel,
             plot_base_name,
             vars_to_plot=None,
             line_style_fn=None,
             plot_names=None,
             save_formats=('png',),
             ylims=None,
             set_xlim_to_min=True,
             use_xmax=None,
             draw_legend=True,
             fig_inches=(18, 12),
             ):

        if vars_to_plot is None:
            vars_to_plot = self.default_vars_to_plot
            ylims = self.default_ylims

        if line_style_fn is None:
            line_style_fn = self.default_line_style_map

        if isinstance(vars_to_plot[0], str):
            vars_to_plot = (vars_to_plot,)

        if plot_names is None:
            plot_names = []
            for x, ys in vars_to_plot:
                xstr = '%s by ' % self.x_names_2_labels[x]
                ystr = ', '.join([self.y_names_2_labels[y][0] for y in ys])
                plot_names.append(xstr+ystr)
        elif isinstance(plot_names, str):
            plot_names = [plot_names,]

        assert(len(vars_to_plot) == len(plot_names))

        # for each plot (i.e., a tuple in vars_to_plot)
        for plot_vars, plot_name in zip(vars_to_plot, plot_names):
            if len(plot_vars) != 2:
                raise ValueError('incorrect vars_to_plot format.  offending vars: %s' % vars)

            full_plot_name = '%s -- %s' % (plot_base_name, plot_name)

            xvar_name = plot_vars[0]
            yvar_names = plot_vars[1]

            # this is not the cleanest way of doing this but i'm making the minimal change here to get plotter working
            # with dataframes since they load so much faster than the full model or even small model files i was using
            # before.  would be better to go back and rewrite all Plotters to work with channel DataFrames from the start
            all_xvals = panel.minor_xs(xvar_name)

            n_subplots = len(yvar_names)
            if ylims is None:
                ylims = [None] * n_subplots

            fig, axes = pylab.subplots(n_subplots, sharex=True)
            if n_subplots == 1:
                axes = (axes,)

            fig.canvas.set_window_title(full_plot_name)
            fig.set_size_inches(*fig_inches)

            minx = np.inf
            maxx = -np.inf

            # for each subplot in this plot
            for yvar_name, ax, ylim in zip(yvar_names, axes, ylims):
                # if yvar_name in panel.minor_xs
                if yvar_name not in panel.minor_axis:
                    # TODO: unhack this once next set of runs fixes name
                    if yvar_name == 'grad_norm' and 'grad_norm_running_avg' in panel.minor_axis:
                        yvar_name = 'grad_norm_running_avg'
                    else:
                        raise ValueError('yvar_name was %s but must be one of: %s' %(yvar_name, '\n'.join(panel.minor_axis)))
                all_yvals = panel.minor_xs(yvar_name)
                if yvar_name == 'grad_norm_running_avg':
                    yvar_name = 'grad_norm'

                yname, use_semilog = self.y_names_2_labels[yvar_name]
                ax.set_ylabel(yname)
                if use_semilog:
                    ax.semilogy()

                if ylim is not None:
                    assert len(ylim) == 2, 'ylim must be an iterable of length 2; got len=%d' % len(ylim)
                    ax.set_ylim(ylim)

                ax.yaxis.grid(b=True, which='major', color='k', linestyle=':', linewidth=1)

                # for each line in this subplot
                for i, data_dir_name in enumerate(all_xvals):
                    x = all_xvals[data_dir_name].values
                    y = all_yvals[data_dir_name].values

                    c, style, width, alpha, zorder, label = line_style_fn(data_dir_name)

                    print 'data_dir_name, label:'
                    print data_dir_name
                    print label
                    print

                    minx = min(minx, x[-1])
                    maxx = max(maxx, x[-1])

                    if np.all(np.bitwise_or(np.isnan(x), np.isnan(y))):
                        continue

                    this_line = ax.plot(x, y, color=c, linestyle=style, zorder=zorder,
                                        alpha=alpha, label=label, linewidth=width)

            xlabel = self.x_names_2_labels[xvar_name]
            axes[-1].set_xlabel(xlabel)

            xlim = minx if set_xlim_to_min else maxx
            xlim = use_xmax if use_xmax is not None else xlim
            axes[-1].set_xlim([0, xlim])

            if draw_legend:
                for ax in axes:
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * self.legend_width_mx, box.height])

                ax = axes[0]

                handles, labels = ax.get_legend_handles_labels()
                # sort both labels and handles by labels
                labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

                # legend
                which_axis = int(floor(len(axes)/2))
                axes[which_axis].legend(handles,
                                        labels,
                                        bbox_to_anchor=(1, 0.5),
                                        loc='center left')

            if save_formats is not None:
                for fmt in save_formats:
                    save_file = os.path.join(self.output_path, full_plot_name+'.'+fmt)
                    print 'Saving:', save_file
                    fig.savefig(save_file, bbox_inches='tight', dpi=300)
        print 'TimeSeriesPlotter: Plots are up'
        pylab.show()


if __name__ == '__main__':

    # exp_file = os.path.join(this_path,
    #                         'output/experiments/cifar_ad----2015.05.26.txt')
    # ylims = [[.22, .35], [1e-2, 1e1], None, None]
    # xmax = 5000
    # vars_to_plot = (('seconds_seen', ['test_y_misclass', 'train_objective']),)
    # p = TimeSeriesPlotter()
    # p.plot_from_experiment(exp_file,
    #                        vars_to_plot=vars_to_plot,
    #                        ylims=ylims,
    #                        use_xmax=xmax,
    #                        fig_inches=(18, 12))

    # exp_file = os.path.join(this_path,
    #                         'output/experiments/cifar_adl----2015.05.26.txt')
    # ylims = [[.22, .35], [1e-2, 1e1], None, None]
    # xmax = 5000
    # vars_to_plot = (('seconds_seen', ['test_y_misclass', 'train_objective']),)
    # p = TimeSeriesPlotter()
    # p.plot_from_experiment(exp_file,
    #                        vars_to_plot=vars_to_plot,
    #                        ylims=ylims,
    #                        use_xmax=xmax,
    #                        fig_inches=(18, 12))

    # exp_file = os.path.join(this_path,
    #                         'output/experiments/cifar_lc----2015.05.26.txt')
    # ylims = [[.22, .35], [1e-2, 1e1], None, None]
    # xmax = 5000
    # vars_to_plot = (('seconds_seen', ['test_y_misclass', 'train_objective']),)
    # p = TimeSeriesPlotter()
    # p.plot_from_experiment(exp_file,
    #                        vars_to_plot=vars_to_plot,
    #                        ylims=ylims,
    #                        use_xmax=xmax,
    #                        fig_inches=(18, 12))

    # exp_file = os.path.join(this_path,
    #                         'output/experiments/cifar_agl----2015.05.26.txt')
    # ylims = [[.22, .35], [1e-2, 1e1], None, None]
    # xmax = 5000
    # vars_to_plot = (('seconds_seen', ['test_y_misclass', 'train_objective']),)
    # p = TimeSeriesPlotter()
    # p.plot_from_experiment(exp_file,
    #                        vars_to_plot=vars_to_plot,
    #                        ylims=ylims,
    #                        use_xmax=xmax,
    #                        fig_inches=(18, 12))

    # exp_file = os.path.join(this_path,
    #                         'output/experiments/cifar_ad-adl-agl----2015.05.26.txt')
    # ylims = [[.22, .35], None, None, None]
    # xmax = 5000
    # vars_to_plot = (('seconds_seen', ['test_y_misclass', 'train_objective']),)
    # p = TimeSeriesPlotter()
    # p.plot_from_experiment(exp_file,
    #                        vars_to_plot=vars_to_plot,
    #                        ylims=ylims,
    #                        use_xmax=xmax,
    #                        fig_inches=(18, 12))

    # exp_file = os.path.join(this_path,
    #                         'output/experiments/cifar_ad-adl----2015.05.26.txt')
    # ylims = [[.22, .35], None, None, None]
    # xmax = 5000
    # vars_to_plot = (('seconds_seen', ['test_y_misclass', 'train_objective']),)
    #
    # p = TimeSeriesPlotter()
    # p.plot_from_experiment(exp_file,
    #                        vars_to_plot=vars_to_plot,
    #                        ylims=ylims,
    #                        use_xmax=xmax,
    #                        fig_inches=(18, 12))

    # exp_file = os.path.join(this_path,
    #                         'output/experiments/cifar_ag-agl----2015.05.26.txt')
    # ylims = [[.2, .5], None, None, None]
    # xmax = 2500
    # vars_to_plot = (('seconds_seen', ['test_y_misclass', 'train_objective', 'learning_rate',  'grad_norm']),)
    #
    # p = TimeSeriesPlotter()
    # p.plot_from_experiment(exp_file,
    #                        vars_to_plot=vars_to_plot,
    #                        ylims=ylims,
    #                        use_xmax=xmax,
    #                        fig_inches=(18, 12))

    # exp_file = os.path.join(this_path,
    #                         'output/experiments/cifar_all_adaptive----2015.05.26.txt')
    # ylims = [[.2, .35], None, None, None]
    # xmax = 2500
    # vars_to_plot = (('seconds_seen', ['test_y_misclass', 'train_objective', 'learning_rate',  'grad_norm']),)
    #
    # p = TimeSeriesPlotter()
    # p.plot_from_experiment(exp_file,
    #                        vars_to_plot=vars_to_plot,
    #                        ylims=ylims,
    #                        use_xmax=xmax,
    #                        fig_inches=(18, 12))
