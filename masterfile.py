##############################################################################################################################################################
##############################################################################################################################################################
# IMPORTS
# from IPython.core.debugger import set_trace
import numpy as np
from numpy.random import RandomState
import scipy as sp
from scipy import ndimage
from random import shuffle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm  # makes loops show smart progress meter

import pickle
import h5py
import copy
import timeit
import time
import datetime
import sys
import os
import shutil
from distutils.dir_util import copy_tree
import importlib

import libraries.inputs as inputs
import libraries.loadsave as io
import libraries.plots as plots
import libraries.functions as fn
import init
import click

from utils import *

def read_from_yaml(file_path, Loader=None):
    """
    Import a python dict from .yaml
    :param file_path: str (should end in '.yaml')
    :param Loader: :class:'yaml.Loader'
    :return: dict
    """
    import yaml
    if Loader is None:
        Loader = yaml.FullLoader
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            data = yaml.load(stream, Loader=Loader)
        return data
    else:
        raise Exception('File: {} does not exist.'.format(file_path))


@click.command()
@click.option('--config_file_path', type=click.Path(exists=True), required=False, default='config/orig_config.yaml')
@click.option('--train', type=bool, default=True)
@click.option('--load_data',is_flag=True)
@click.option('--data_file_path', type=click.Path(exists=True), required=False)
@click.option('--plot', is_flag=True)
@click.option('--test', is_flag=True)
@click.option('--export', is_flag=True)
@click.option('--export_file_name',type=click.Path(exists=False), required=False, default=None)
@click.option('--export_dir_path',type=click.Path(exists=True), required=False, default='data')
@click.option('--debug', is_flag=True)
@click.option('--interactive',is_flag=True)

def main(config_file_path, train, load_data, data_file_path, plot, test, export, export_file_name, export_dir_path,
         debug,interactive):
    """

    :param config_file_path:
    :param train:
    :param load_data:
    :param data_file_path:
    :param plot:
    :param test:
    :param export:
    :param export_file_name:
    :param export_dir_path:
    :param debug:
    """

    config_dict = read_from_yaml(config_file_path)
    network = Network(**config_dict, export=export, export_dir_path=export_dir_path, export_file_name=export_file_name)

    if debug:
        return

    np.seterr(all='raise')  # raise error in case of RuntimeWarning
    network.network_states[0] = copy.deepcopy([0] + network.network)  # save network state right after initialization at t=0

    #SIM

    if train:
        network.train()

    if load_data:
        network.load_data(data_file_path)

    if test:
        network.test()

    if plot:
        network.plot_graphs()

    if interactive:
        globals().update(locals())

if __name__ == '__main__':
    main()