
import numpy as np
import matplotlib;
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from random import shuffle

def get_input(n, **keyword_parameters):

    # input orientation
    if ('theta' in keyword_parameters):
        s=keyword_parameters['theta']
    else:
        s = np.random.rand()*np.pi  # set orientation randomly (uniform sample)
        if ('symmetry' in keyword_parameters):
            if keyword_parameters['symmetry'] == False:
                mean = np.pi/2
                stdv = np.pi/3
                s = np.random.normal(mean, stdv) % np.pi

    s_y = np.arange(0, n, 1) * (np.pi/n)  # center of L4 tuning curves (i.e. most responsive stimulus)

    # define tuning curves: normal distributions
    d = np.minimum(np.minimum(np.abs(s-s_y), np.abs(s-(s_y+np.pi))), np.abs(s-(s_y-np.pi))) # get minimum distance to tuning curve peak (due to ring structure of stimulus space)

    # width of tuning curves
    if ('sigma' in keyword_parameters):
        sigma = keyword_parameters['sigma']
    else:
        sigma = 20/360 * 2*np.pi

    # response amplitude
    if ('amp' in keyword_parameters):
        amp = keyword_parameters['amp']
    else:
        amp = 40

    y = amp*np.exp(-(d**2)/(2*sigma**2))

    # simultaneously present an additional orthogonal orientation
    if ('binoc' in keyword_parameters):
        if keyword_parameters['binoc'] != 0:
            binoc = keyword_parameters['binoc']
            s_binoc = s + binoc
            d_binoc = np.minimum(np.minimum(np.abs(s_binoc-s_y), np.abs(s_binoc-(s_y+np.pi))), np.abs(s_binoc-(s_y-np.pi))) # get minimum distance to tuning curve peak (due to ring structure of stimulus space)
            y_binoc = amp*np.exp(-(d_binoc**2)/(2*sigma**2))
            y += y_binoc
        else:
            y_binoc = 0
    else:
        y_binoc = 0

    return y
