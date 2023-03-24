
import numpy as np
from numpy.random import RandomState
# from numba import jit
import scipy as sp
from scipy import ndimage
from random import shuffle
import matplotlib;
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import copy
import timeit
import time
import sys
import os
import shutil
from distutils.dir_util import copy_tree


def make_cmap(colors, position=None, bit=False):

    # make_cmap takes a list of tuples which contain RGB values. The RGB
    # values may either be in 8-bit [0 to 255] (in which bit must be set to
    # True when called) or arithmetic [0 to 1] (default). make_cmap returns
    # a cmap with equally spaced colors.
    # Arrange your tuples so that the first color is the lowest value for the
    # colorbar and the last is the highest.
    # position contains values from 0 to 1 to dictate the location of each color.

    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


# extract weights from network state recordings
def extract_weight_hist(network_states, ind, **kwargs):
    if ('downSampleRatio' in kwargs):
        downSampleRatio = kwargs['downSampleRatio']
    else:
        downSampleRatio = 10
    N_t = len(network_states) # number of datapoints per weight
    if ind != 0:
        [N_post, N_pre] = network_states[0][ind].shape
        if N_pre != 0:
            W = np.zeros((N_t, N_post, N_pre))
            for t in range(N_t): # cycle through saved timepoints
                W[t, :, :] = network_states[t][ind] # extract weights
        return W[::downSampleRatio,:]
    elif ind == 0:
        sample_time = np.zeros(N_t)
        for t in range(N_t): # cycle through saved timepoints
            sample_time[t] = network_states[t][ind]
        return sample_time[::downSampleRatio]


# returns circular gaussian distribution
def gaussian(x, y, sigma, amp=1, **kwargs):
    d = cyclic_distance(x,y)
    gauss = np.exp(-d**2/(2*sigma**2))

    if ('norm' in kwargs):
        if kwargs['norm']:
            amp /= np.sum(gauss, axis=1)[:, np.newaxis]

    return amp * gauss


# returns minimum distance between two tuning peaks in degree
def cyclic_distance(x, y):
    x =  x * np.ones((1,))
    y =  y * np.ones((1,))
    return np.minimum(np.minimum(np.abs(x[:,None]-y[None,:]), np.abs(x[:,None]-(y[None,:]+180))), np.abs(x[:,None]-(y[None,:]-180)))
