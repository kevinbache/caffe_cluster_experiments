#!/usr/bin/env python

"""
Parse training log to get loss values / gradient norms / learning rates.
This is a modified version of parse_log.py that is released by Caffe
"""
import sys
import os
import re
import argparse
import pandas as pd
import numpy as np

# append this path for extract_seconds
this_path = os.path.dirname(os.path.realpath(__file__))
sys.path += [this_path]

import extract_seconds

def get_line_type(line):
    """Return either 'test' or 'train' depending on line type
    """

    line_type = None
    if line.find('Train') != -1:
        line_type = 'train'
    elif line.find('Test') != -1:
        line_type = 'test'
    return line_type


def parse_log(path_to_log):
    """Parse log file
    Returns (train_dict_list, train_dict_names, test_dict_list, test_dict_names)

    train_dict_list and test_dict_list are lists of dicts that define the table
    rows

    train_dict_names and test_dict_names are ordered tuples of the column names
    for the two dict_lists
    """

    re_batch_size = re.compile('batch_size: (\d+)')
    re_iteration = re.compile('Iteration (\d+)')
    re_accuracy = re.compile('output #\d+: accuracy = ([\.\d\-+e]+)')
    re_train_loss = re.compile('Iteration \d+, loss = ([\.\d\-+e]+)')
    re_output_loss = re.compile('output #\d+: loss = ([\.\d\-+e]+)')
    re_lr = re.compile('lr = ([\.\d\-+e]+)')
    re_grad_norm = re.compile('avg_grad_norm = ([\.\d\-+e]+)')
    re_test_start_seconds = re.compile('Testing net')

    # Pick out lines of interest
    iteration = -1
    test_accuracy = -1
    test_start_seconds = float('NaN')
    learning_rate = float('NaN')
    avg_grad_norm = float('NaN')
    batch_size = None
    train_dict_list = []
    test_dict_list = []

    logfile_year = extract_seconds.get_log_created_year(path_to_log)
    with open(path_to_log) as f:
        start_time = extract_seconds.get_start_time(f, logfile_year)

    with open(path_to_log) as f:
        for line in f:
            if batch_size is None:
                batch_size_match = re_batch_size.search(line)
                if batch_size_match:
                    batch_size = float(batch_size_match.group(1))

            iteration_match = re_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))

            if iteration == -1:
                # Only look for other stuff if we've found the first iteration
                continue

            time = extract_seconds.extract_datetime_from_line(line,
                                                              logfile_year)
            seconds = (time - start_time).total_seconds()

            lr_match = re_lr.search(line)
            if lr_match:
                learning_rate = float(lr_match.group(1))

            grad_norm_match = re_grad_norm.search(line)
            if grad_norm_match:
                avg_grad_norm = float(grad_norm_match.group(1))

            test_start_match = re_test_start_seconds.search(line)
            if test_start_match:
                test_start_seconds = seconds

            accuracy_match = re_accuracy.search(line)
            if accuracy_match and get_line_type(line) == 'test':
                test_accuracy = float(accuracy_match.group(1))

            train_loss_match = re_train_loss.search(line)
            if train_loss_match:
                train_loss = float(train_loss_match.group(1))
                train_dict_list.append({'NumIters': iteration,
                                        'Seconds': seconds,
                                        'TrainingLoss': train_loss,
                                        'LearningRate': learning_rate,
                                        'AvgGradientNorm': avg_grad_norm})

            output_loss_match = re_output_loss.search(line)
            if output_loss_match and get_line_type(line) == 'test':
                test_loss = float(output_loss_match.group(1))
                # NOTE: we assume that (1) accuracy always comes right before
                # loss for test data so the test_accuracy variable is already
                # correctly populated and (2) there's one and only one output
                # named "accuracy" for the test net
                test_dict_list.append({'NumIters': iteration,
                                       'SecondsAtStart': test_start_seconds,
                                       'SecondsAtEnd': seconds,
                                       'TestAccuracy': test_accuracy,
                                       'TestLoss': test_loss})

    return train_dict_list, test_dict_list, batch_size


def parse_args():
    description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')

    args = parser.parse_args()
    return args

def create_dataframe(train_dict_list, test_dict_list, batch_size):
    df_extra = pd.DataFrame(train_dict_list)

    # this assumes that the ouput from the test-on-train net comes before the test-on-test net
    df_train = pd.DataFrame(test_dict_list[0::2])
    df_test = pd.DataFrame(test_dict_list[1::2])

    total_test_time = df_train['SecondsAtEnd'] - df_train['SecondsAtStart']
    total_test_time += df_test['SecondsAtEnd'] - df_test['SecondsAtStart']

    total_test_time = np.cumsum(total_test_time)

    df = pd.DataFrame()
    df['batches_seen'] = df_test['NumIters']
    df.index.name = 'epochs_seen'
    if batch_size is not None:
        df['datapoints_seen'] = df_test['NumIters'] * batch_size
    df['seconds_seen'] = df_test['SecondsAtEnd'] - total_test_time
    df['seconds_seen_test'] = total_test_time

    df['train_accuracy'] = df_train['TestAccuracy']
    df['train_y_misclass'] = 1 - df['train_accuracy']
    df['train_objective'] = df_train['TestLoss']

    df['test_accuracy'] = df_test['TestAccuracy']
    df['test_y_misclass'] = 1 - df['test_accuracy']
    df['test_objective'] = df_test['TestLoss']

    # get the gradient norm / learning rate statistics
    iters_per_test = df['batches_seen'][1]  # the 0th entry will always be 0, get the 1st
    eps = 0.001 # this is just to make the int division below work correctly
    # the + 1 here is so there's a nan at the start of each series below
    # (e.g.: df['LearningRateLast']) instead of at the end. it works because df_extra['TestEpoch']
    # will get treated like an index by pandas in the groupby operation
    df_extra['TestEpoch'] = ((df_extra['NumIters'] - eps) / iters_per_test).astype('int') + 1

    gb_extra = df_extra.groupby('TestEpoch', as_index=True)

    df['learning_rate_last'] = gb_extra['LearningRate'].last()
    df['learning_rate'] = gb_extra['LearningRate'].mean()
    df['grad_norm_last'] = gb_extra['AvgGradientNorm'].last()
    df['grad_norm'] = gb_extra['AvgGradientNorm'].mean()
    df['training_displays_count'] = gb_extra['LearningRate'].count()

    return df


def save_df(out_file, df):
    if hasattr(df, 'to_pickle'):
        # true for recent pandas
        df.to_pickle(out_file)
    else:
        # true for old pandas
        df.save(out_file)


def main(logfile_path, verbose):
    train_dict_list, test_dict_list, batch_size = parse_log(logfile_path)
    df = create_dataframe(train_dict_list, test_dict_list, batch_size)

    if verbose:
        print df

    out_dir = os.path.dirname(logfile_path)
    out_file = os.path.join(out_dir, 'parsed_log_df.pkl')
    save_df(out_file, df)

if __name__ == '__main__':
    args = parse_args()
    main(args.logfile_path, args.verbose)
