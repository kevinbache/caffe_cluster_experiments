__author__ = 'kevin'

def load_channels(channels_filename):
    """
    load a channels_filename data frame saved by save_channels

    :param channels_filename: the name of the channels_filename file to load
    :rtype: pandas.DataFrame
    :return: DataFrame representing all the channels_filename in a model, time slice is the index and channel names are
             the columns and the name of the algorithm that generated this run, i.e. the name of the directory in which
             the channels file is contained
    """
    import os
    import pandas as pd
    if hasattr(pd, 'read_pickle'):
        # true for recent pandas
        df = pd.read_pickle(channels_filename)
    else:
        # true for old pandas
        df = pd.load(channels_filename)
    algorithm_name = os.path.basename(os.path.dirname(channels_filename))
    return df, algorithm_name
