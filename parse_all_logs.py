import argparse

__author__ = 'kevin'
import os
import sys

# append this path for extract_seconds
this_path = os.path.dirname(os.path.realpath(__file__))
sys.path += [this_path]

import parse_log

default_root = os.path.join(this_path, 'output', 'runs')

def find_files_recursive(root_path, pattern):
    """
    find all files mathching a partern recursively starting at a root path

    :param root_path: look in all directories rooted at this one
    :param pattern: the fnmatch pattern to apply to the name of each file
    :return: a list of all matching file names
    """
    import fnmatch
    import os

    matches = []
    for root, dirnames, filenames in os.walk(root_path):
      for filename in fnmatch.filter(filenames, pattern):
          matches.append(os.path.join(root, filename))

    return matches

def parse_my_args():
    description = ('Parse all Caffe training logs rooted at root_path')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--root_path',
                        default = default_root,
                        help='Path to recursively search for error.log files to parse.')

    parser.add_argument('--pattern',
                        default = 'error.log',
                        help='Pattern to search for')


    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info ')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_my_args()
    files = find_files_recursive(args.root_path, 'error.log')
    for f in files:
        print 'Starting:', f
        parse_log.main(f, args.verbose)

