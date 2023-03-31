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
@click.option('--data_file_path', type=click.Path(exists=True), required=False)
@click.option('--plot', is_flag=True)
@click.option('--run_analysis', is_flag=True)
@click.option('--export', is_flag=True)
@click.option('--debug', is_flag=True)
@click.option('--run_sim', type=bool, default=True)
def main(config_file_path, data_file_path, plot, run_analysis, export, debug, run_sim):

    config_dict = read_from_yaml(config_file_path)
    init.update_namespace(config_dict)

    short_description = init.short_description
    random_seed = init.random_seed

    online_saves = init.online_saves

    T = init.T
    T_seq = init.T_seq
    T_fine = init.T_fine

    N_pc = init.N_pc
    N_pv = init.N_pv
    N_L4 = init.N_L4

    gain = init.gain
    bias = init.bias
    n_exp = init.n_exp

    W_EE_norm = init.W_EE_norm
    W_EI_norm = init.W_EI_norm
    W_IE_norm = init.W_IE_norm
    W_II_norm = init.W_II_norm
    W_EF_norm = init.W_EF_norm
    W_IF_norm = init.W_IF_norm

    e_x_pc = 1/init.e_x_pc_inv
    e_x_pv = 1/init.e_x_pv_inv
    e_y = init.e_y

    std_W_EL = init.std_W_EL
    std_W_IL = init.std_W_IL
    std_W_II = init.std_W_II
    std_W_EI = init.std_W_EI
    std_W_IE = init.std_W_IE
    std_W_EE = init.std_W_EE

    tuned_init = init.tuned_init
    sigma_ori = init.sigma_ori

    input_sigma = init.input_sigma
    input_amp = init.input_amp

    l_norm = init.l_norm
    joint_norm = init.joint_norm
    lateral_norm = init.lateral_norm

    e = init.e

    e_w = init.e_w
    e_k = init.e_k
    e_p = init.e_p
    e_m = init.e_m
    e_q = init.e_q
    e_u = init.e_u

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d--%H-%M-%S')

    # -----------------------

    save_path = "data/" + timestamp + "--" + short_description.replace(" ", "_")
    save_path_figures = save_path + '/figures'
    save_path_network = save_path + ''
    save_path_libraries = save_path + '/libraries'

    for path in [save_path, save_path_figures, save_path_network, save_path_libraries]:
        fn.create_directory(path)

    shutil.copy2("sim.py", save_path + "/sim.py")
    shutil.copy2("init.py", save_path + "/init.py")
    shutil.copy2("config.py", save_path + "/config.py")
    shutil.copy2("libraries/inputs.py", save_path_libraries + "/inputs.py")
    shutil.copy2("libraries/functions.py", save_path_libraries + "/functions.py")
    shutil.copy2("libraries/loadsave.py", save_path_libraries + "/loadsave.py")
    shutil.copy2("libraries/plots.py", save_path_libraries + "/plots.py")

    ##############################################################################

    np.random.seed(random_seed)

    ###############################################################################
    accuracy = int(np.ceil(T / 10 ** 4))  # monitor recording variables every 'accuracy' iterations,
    # i.e., 10**x sample points in total
    accuracy_states = accuracy * 40  # monitor full network network every 'accuracy_states' iterations

    e_w_EE = e * 0.10 * 10 ** (-7)  # weight learning rate for excitatory connections onto excitatory neurons
    e_w_EI = e * 0.20 * 10 ** (-7)  # weight learning rate for inhibitory connections onto excitatory neurons

    e_w_IE = e * 0.15 * 10 ** (-7)  # weight learning rate for excitatory connections onto inhibitory neurons
    e_w_II = e * 0.25 * 10 ** (-7)  # weight learning rate for inhibitory connections onto inhibitory neurons

    # preallocate

    # L4 input
    y_L4 = np.zeros((N_L4,))  # L4 input
    y_L4_mean_init = 0.1
    y_mean_L4 = np.zeros((N_L4,)) + y_L4_mean_init

    # L2/3 PYR principal cells (pc)
    y_pc = np.zeros((N_pc,))
    y_0_pc = np.zeros((N_pc,)) + 1
    y_mean_pc = np.ones((N_pc,)) * 1
    y_var_pc = np.ones((N_pc,)) * 0.1
    x_pc = np.zeros((N_pc,))
    x_e_pc = np.zeros((N_pc,))
    x_i_pc = np.zeros((N_pc,))
    x_e_mean_pc = np.zeros((N_pc,))
    x_i_mean_pc = np.zeros((N_pc,))
    x_mean_pc = np.zeros((N_pc,))
    a_pc = np.zeros((N_pc,)) + gain
    b_pc = np.zeros((N_pc,)) + bias
    n_pc = np.zeros((N_pc,)) + n_exp
    W_EF_norms = np.zeros((N_pc,)) + W_EF_norm
    W_EI_norms = np.zeros((N_pc,)) + W_EI_norm
    W_EE_norms = np.zeros((N_pc,)) + W_EE_norm

    # L2/3 PV inhibitory cells (pv)
    y_pv = np.zeros((N_pv,))
    y_0_pv = np.zeros((N_pv,)) + 1
    y_mean_pv = np.ones((N_pv,)) * 1
    y_var_pv = np.ones((N_pv,)) * 0.1
    x_pv = np.zeros((N_pv,))
    x_e_pv = np.zeros((N_pv,))
    x_i_pv = np.zeros((N_pv,))
    x_e_mean_pv = np.zeros((N_pv,))
    x_i_mean_pv = np.zeros((N_pv,))
    x_mean_pv = np.zeros((N_pv,))
    a_pv = np.zeros((N_pv,)) + gain
    b_pv = np.zeros((N_pv,)) + bias
    n_pv = np.zeros((N_pv,)) + n_exp
    W_IF_norms = np.zeros((N_pv,)) + W_IF_norm
    W_II_norms = np.zeros((N_pv,)) + W_II_norm
    W_IE_norms = np.zeros((N_pv,)) + W_IE_norm

    # initialize weights
    W = abs(np.random.normal(2 * std_W_EL, std_W_EL, (N_pc, N_L4)))  # (L4->PC)
    K = abs(np.random.normal(2 * std_W_IL, std_W_IL, (N_pv, N_L4)))  # (L4->PV)
    P = abs(np.random.normal(2 * std_W_II, std_W_II, (N_pv, N_pv)))  # (PV-|PV)
    M = abs(np.random.normal(2 * std_W_EI, std_W_EI, (N_pc, N_pv)))  # (PV-|PC)
    Q = abs(np.random.normal(2 * std_W_IE, std_W_IE, (N_pv, N_pc)))  # (PC->PV)
    U = abs(np.random.normal(2 * std_W_EE, std_W_EE, (N_pc, N_pc)))  # (PC->PC)

    ###############################################################################
    # history preallocation

    network_states = [[]] * (2 + int(T / accuracy_states))  # preallocate memory to track network state
    count = int(T / accuracy)  # number of entries in history arrays
    t_hist = np.zeros((count,))

    # recorded from all neurons

    # PC
    x_pc_hist = np.zeros((count, N_pc))
    x_e_pc_hist = np.zeros((count, N_pc))
    x_i_pc_hist = np.zeros((count, N_pc))
    x_e_mean_pc_hist = np.zeros((count, N_pc))
    x_i_mean_pc_hist = np.zeros((count, N_pc))
    b_pc_hist = np.zeros((count, N_pc))
    a_pc_hist = np.zeros((count, N_pc))
    n_pc_hist = np.zeros((count, N_pc))
    W_EF_norms_hist = np.zeros((count, N_pc))
    W_EI_norms_hist = np.zeros((count, N_pc))
    W_EE_norms_hist = np.zeros((count, N_pc))
    y_pc_hist = np.zeros((count, N_pc))
    y_mean_pc_hist = np.zeros((count, N_pc))
    y_var_pc_hist = np.zeros((count, N_pc))
    y_0_pc_hist = np.zeros((count, N_pc))

    # PV
    x_pv_hist = np.zeros((count, N_pv))
    x_e_pv_hist = np.zeros((count, N_pv))
    x_i_pv_hist = np.zeros((count, N_pv))
    x_e_mean_pv_hist = np.zeros((count, N_pv))
    x_i_mean_pv_hist = np.zeros((count, N_pv))
    b_pv_hist = np.zeros((count, N_pv))
    a_pv_hist = np.zeros((count, N_pv))
    n_pv_hist = np.zeros((count, N_pv))
    W_IF_norms_hist = np.zeros((count, N_pv))
    W_II_norms_hist = np.zeros((count, N_pv))
    W_IE_norms_hist = np.zeros((count, N_pv))
    y_pv_hist = np.zeros((count, N_pv))
    y_mean_pv_hist = np.zeros((count, N_pv))
    y_var_pv_hist = np.zeros((count, N_pv))
    y_0_pv_hist = np.zeros((count, N_pv))

    y_mean_L4_hist = np.zeros((count, N_L4))

    # recorded from one neuron
    W_hist = np.zeros((count, N_L4))  # (L4->PC)
    K_hist = np.zeros((count, N_L4))  # (L4->PV)
    P_hist = np.zeros((count, N_pv))  # (PV-|PV)
    M_hist = np.zeros((count, N_pv))  # (PV-|PC)
    Q_hist = np.zeros((count, N_pc))  # (PC->PV)
    U_hist = np.zeros((count, N_pc))  # (PC->PC)

    # recorded from all neurons
    W_norm_hist = np.zeros((count, N_pc))  # (L4->PC)
    K_norm_hist = np.zeros((count, N_pv))  # (L4->PV)
    P_norm_hist = np.zeros((count, N_pv))  # (PV-|PV)
    M_norm_hist = np.zeros((count, N_pc))  # (PV-|PC)
    Q_norm_hist = np.zeros((count, N_pv))  # (PC->PV)
    U_norm_hist = np.zeros((count, N_pc))  # (PC->PC)

    # record variables every iteration during last 'T_fine' iterations

    # PC
    x_pc_hist_fine = np.zeros((T_fine, N_pc))
    x_e_pc_hist_fine = np.zeros((T_fine, N_pc))
    x_i_pc_hist_fine = np.zeros((T_fine, N_pc))
    x_e_mean_pc_hist_fine = np.zeros((T_fine, N_pc))
    x_i_mean_pc_hist_fine = np.zeros((T_fine, N_pc))
    b_pc_hist_fine = np.zeros((T_fine, N_pc))
    a_pc_hist_fine = np.zeros((T_fine, N_pc))
    n_pc_hist_fine = np.zeros((T_fine, N_pc))
    W_EF_norms_hist_fine = np.zeros((T_fine, N_pc))
    W_EI_norms_hist_fine = np.zeros((T_fine, N_pc))
    W_EE_norms_hist_fine = np.zeros((T_fine, N_pc))
    y_pc_hist_fine = np.zeros((T_fine, N_pc))
    y_mean_pc_hist_fine = np.zeros((T_fine, N_pc))
    y_var_pc_hist_fine = np.zeros((T_fine, N_pc))
    y_0_pc_hist_fine = np.zeros((T_fine, N_pc))

    # PV
    x_pv_hist_fine = np.zeros((T_fine, N_pv))
    x_e_pv_hist_fine = np.zeros((T_fine, N_pv))
    x_i_pv_hist_fine = np.zeros((T_fine, N_pv))
    x_e_mean_pv_hist_fine = np.zeros((T_fine, N_pv))
    x_i_mean_pv_hist_fine = np.zeros((T_fine, N_pv))
    b_pv_hist_fine = np.zeros((T_fine, N_pv))
    a_pv_hist_fine = np.zeros((T_fine, N_pv))
    n_pv_hist_fine = np.zeros((T_fine, N_pv))
    W_IF_norms_hist_fine = np.zeros((T_fine, N_pv))
    W_II_norms_hist_fine = np.zeros((T_fine, N_pv))
    W_IE_norms_hist_fine = np.zeros((T_fine, N_pv))
    y_pv_hist_fine = np.zeros((T_fine, N_pv))
    y_mean_pv_hist_fine = np.zeros((T_fine, N_pv))
    y_var_pv_hist_fine = np.zeros((T_fine, N_pv))
    y_0_pv_hist_fine = np.zeros((T_fine, N_pv))

    y_L4_hist_fine = np.zeros((T_fine, N_L4))
    y_mean_L4_hist_fine = np.zeros((T_fine, N_L4))

    # recorded from one neuron
    W_hist_fine = np.zeros((T_fine, N_pc, N_L4))  # (L4->PC)
    K_hist_fine = np.zeros((T_fine, N_pv, N_L4))  # (L4->PV)
    P_hist_fine = np.zeros((T_fine, N_pv, N_pv))  # (PV-|PV)
    M_hist_fine = np.zeros((T_fine, N_pc, N_pv))  # (PV-|PC)
    Q_hist_fine = np.zeros((T_fine, N_pv, N_pc))  # (PC->PV)
    U_hist_fine = np.zeros((T_fine, N_pc, N_pc))  # (PC->PC)

    # initialize tuned weights
    if tuned_init:
        d_theta_in = 180 / N_L4
        d_theta_pc = 180 / N_pc
        d_theta_pv = 180 / N_pv
        theta_pc = np.arange(0, 180, d_theta_pc)  # tuning peaks of pc neurons
        theta_pv = np.arange(0, 180, d_theta_pv)  # tuning peaks of pv neurons
        theta_in = np.arange(0, 180, d_theta_in)  # tuning peaks of in neurons

        W = fn.gaussian(theta_pc, theta_in, input_sigma)  # weight connectivity kernels
        K = fn.gaussian(theta_pv, theta_in, input_sigma)
        U = fn.gaussian(theta_pc, theta_pc, sigma_ori)
        Q = fn.gaussian(theta_pv, theta_pc, sigma_ori)
        M = fn.gaussian(theta_pc, theta_pv, sigma_ori)
        P = fn.gaussian(theta_pv, theta_pv, sigma_ori)

    norm_W = (np.sum(W ** l_norm, axis=1)) ** (1 / l_norm)  # EF
    norm_K = (np.sum(K ** l_norm, axis=1)) ** (1 / l_norm)  # IF
    norm_U = (np.sum(U ** l_norm, axis=1)) ** (1 / l_norm)  # EE
    norm_M = (np.sum(M ** l_norm, axis=1)) ** (1 / l_norm)  # EI
    norm_Q = (np.sum(Q ** l_norm, axis=1)) ** (1 / l_norm)  # IE
    norm_P = (np.sum(P ** l_norm, axis=1)) ** (1 / l_norm)  # II

    # after initialization, normalize input streams

    # first establish feedforward and lateral weight norms as defined in config file (with small lateral E weights)
    W[norm_W != 0, :] *= (W_EF_norms[norm_W != 0, np.newaxis] / norm_W[norm_W != 0, np.newaxis])
    M[norm_M != 0, :] *= (W_EI_norms[norm_M != 0, np.newaxis] / norm_M[norm_M != 0, np.newaxis])
    U[norm_U != 0, :] *= (W_EE_norms[norm_U != 0, np.newaxis] / norm_U[norm_U != 0, np.newaxis])

    K[norm_K != 0, :] *= (W_IF_norms[norm_K != 0, np.newaxis] / norm_K[norm_K != 0, np.newaxis])
    P[norm_P != 0, :] *= (W_II_norms[norm_P != 0, np.newaxis] / norm_P[norm_P != 0, np.newaxis])
    Q[norm_Q != 0, :] *= (W_IE_norms[norm_Q != 0, np.newaxis] / norm_Q[norm_Q != 0, np.newaxis])

    norm_W = (np.sum(W ** l_norm, axis=1)) ** (1 / l_norm)  # EF
    norm_K = (np.sum(K ** l_norm, axis=1)) ** (1 / l_norm)  # IF
    norm_U = (np.sum(U ** l_norm, axis=1)) ** (1 / l_norm)  # EE
    norm_M = (np.sum(M ** l_norm, axis=1)) ** (1 / l_norm)  # EI
    norm_Q = (np.sum(Q ** l_norm, axis=1)) ** (1 / l_norm)  # IE
    norm_P = (np.sum(P ** l_norm, axis=1)) ** (1 / l_norm)  # II

    # normalize all excitatory inputs (lateral and ffwd.) together
    W_EE_norms = W_EE_norms + W_EF_norms
    W_IE_norms = W_IE_norms + W_IF_norms

    norm_WU = (norm_W ** l_norm + norm_U ** l_norm) ** (1 / l_norm)  # EE
    norm_KQ = (norm_K ** l_norm + norm_Q ** l_norm) ** (1 / l_norm)  # IE

    # normalization of exciatory PC input
    W *= (W_EE_norms[:, np.newaxis] / norm_WU[:, np.newaxis])
    U *= (W_EE_norms[:, np.newaxis] / norm_WU[:, np.newaxis])

    # normalization of inhibitory PC input
    M[norm_M != 0, :] *= (
                W_EI_norms[norm_M != 0, np.newaxis] / norm_M[norm_M != 0, np.newaxis])  # do not scale 0-norm input

    # joint normalization of all exciatory PV input
    K *= (W_IE_norms[:, np.newaxis] / norm_KQ[:, np.newaxis])
    Q *= (W_IE_norms[:, np.newaxis] / norm_KQ[:, np.newaxis])

    # normalization of inhibitory PV input
    P[norm_P != 0, :] *= (
                W_II_norms[norm_P != 0, np.newaxis] / norm_P[norm_P != 0, np.newaxis])  # do not scale 0-norm input

    # gather network state variables in a list for later saving
    network = [W, K, P, M, Q, U,
               a_pc, b_pc, n_pc, W_EF_norms, W_EI_norms, W_EE_norms, y_mean_pc, y_var_pc, y_0_pc,
               a_pv, b_pv, n_pv, W_IF_norms, W_II_norms, W_IE_norms, y_mean_pv, y_var_pv, y_0_pv]

    it = iter(np.arange(len(network)) + 1)  # create iterator
    net_dic = {"W": next(it), "K": next(it), "P": next(it), "M": next(it), "Q": next(it), "U": next(it),
               "a\_pc": next(it), "b\_pc": next(it), "n\_pc": next(it), "W\_EF\_norm": next(it),
               "W\_EI\_norm": next(it), "W\_EE\_norm": next(it), "y\_mean\_pc": next(it), "y\_var\_pc": next(it),
               "y\_0\_pc": next(it),
               "a\_pv": next(it), "b\_pv": next(it), "n\_pv": next(it), "W\_IF\_norm": next(it),
               "W\_II\_norm": next(it), "W\_IE\_norm": next(it), "y\_mean\_pv": next(it), "y\_var\_pv": next(it),
               "y\_0\_pv": next(it)}

    hist = [x_pc_hist, x_e_pc_hist, x_i_pc_hist, x_e_mean_pc_hist, x_i_mean_pc_hist, b_pc_hist, a_pc_hist, n_pc_hist,
            W_EF_norms_hist, W_EI_norms_hist, W_EE_norms_hist, y_pc_hist, y_mean_pc_hist, y_var_pc_hist, y_0_pc_hist,
            x_pv_hist, x_e_pv_hist, x_i_pv_hist, x_e_mean_pv_hist, x_i_mean_pv_hist, b_pv_hist, a_pv_hist, n_pv_hist,
            W_IF_norms_hist, W_II_norms_hist, W_IE_norms_hist, y_pv_hist, y_mean_pv_hist, y_var_pv_hist, y_0_pv_hist,
            W_hist, K_hist, P_hist, M_hist, Q_hist, U_hist,
            W_norm_hist, K_norm_hist, P_norm_hist, M_norm_hist, Q_norm_hist, U_norm_hist]

    hist_fine = [x_pc_hist_fine, x_e_pc_hist_fine, x_i_pc_hist_fine, x_e_mean_pc_hist_fine, x_i_mean_pc_hist_fine,
                 b_pc_hist_fine, a_pc_hist_fine, n_pc_hist_fine, W_EF_norms_hist_fine, W_EI_norms_hist_fine,
                 W_EE_norms_hist_fine, y_pc_hist_fine, y_mean_pc_hist_fine, y_var_pc_hist_fine, y_0_pc_hist_fine,
                 x_pv_hist_fine, x_e_pv_hist_fine, x_i_pv_hist_fine, x_e_mean_pv_hist_fine, x_i_mean_pv_hist_fine,
                 b_pv_hist_fine, a_pv_hist_fine, n_pv_hist_fine, W_IF_norms_hist_fine, W_II_norms_hist_fine,
                 W_IE_norms_hist_fine, y_pv_hist_fine, y_mean_pv_hist_fine, y_var_pv_hist_fine, y_0_pv_hist_fine,
                 y_L4_hist_fine,
                 W_hist_fine, K_hist_fine, P_hist_fine, M_hist_fine, Q_hist_fine, U_hist_fine]
    param = [e_x_pc, e_x_pv, e_y, e_w_EE, e_w_IE, e_w_EI, e_w_II,
             N_L4, N_pc, N_pv, input_amp, input_sigma]

    if debug:
        return

    np.seterr(all='raise')  # raise error in case of RuntimeWarning

    network_states[0] = copy.deepcopy([0] + network)  # save network state right after initialization at t=0

    ##############################################################################################################################################################
    ##############################################################################################################################################################
    # SIM

    if run_sim:

        start_time = time.time()
        print_time = 0  # needed for check how much time has passed since the last time progress has been printed

        for t in tqdm(range(T)):

            # load input patch
            if t % T_seq == 0:  # switch input from L4 every 'T_seq' iterations
                if t >= T - 1000:  # in last 1k iterations sweep over whole input space to measure orientation tunings
                    orientation = ((t / T_seq) % 25) / 25 * np.pi  # sweep over all orientation within 25 trials
                    y_L4 = inputs.get_input(N_L4, theta=orientation, sigma=input_sigma * np.pi / 180, amp=input_amp)
                else:
                    y_L4 = inputs.get_input(N_L4, sigma=input_sigma * np.pi / 180, amp=input_amp)

            y_mean_L4 += e_y * (-y_mean_L4 + y_L4)

            x_e_pc = np.dot(W, y_L4) + np.dot(U, y_pc)
            x_i_pc = np.dot(M, y_pv)

            x_e_pv = np.dot(K, y_L4) + np.dot(Q, y_pc)
            x_i_pv = np.dot(P, y_pv)

            x_pc += e_x_pc * (-x_pc + x_e_pc - x_i_pc)
            x_pv += e_x_pv * (-x_pv + x_e_pv - x_i_pv)

            y_pc = a_pc * ((abs(x_pc) + x_pc) / 2) ** n_pc
            y_pv = a_pv * ((abs(x_pv) + x_pv) / 2) ** n_pv

            ###############################################################################

            # running averages and variances
            y_mean_pc_old = y_mean_pc
            y_mean_pc += e_y * (-y_mean_pc + y_pc)
            y_var_pc += e_y * (-y_var_pc + (y_pc - y_mean_pc_old) * (y_pc - y_mean_pc))
            x_e_mean_pc += e_y * (-x_e_mean_pc + x_e_pc)
            x_i_mean_pc += e_y * (-x_i_mean_pc + x_i_pc)

            y_mean_pv_old = y_mean_pv
            y_mean_pv += e_y * (-y_mean_pv + y_pv)
            y_var_pv += e_y * (-y_var_pv + (y_pv - y_mean_pv_old) * (y_pv - y_mean_pv))
            x_e_mean_pv += e_y * (-x_e_mean_pv + x_e_pv)
            x_i_mean_pv += e_y * (-x_i_mean_pv + x_i_pv)

            # weight adaption
            W += e_w * e_w_EE * np.outer(y_pc, y_L4)  # L4->PC
            K += e_k * e_w_IE * np.outer(y_pv, y_L4)  # L4->PV
            P += e_p * e_w_II * np.outer(y_pv, y_pv)  # PV-|PV
            M += e_m * e_w_EI * np.outer(y_pc, y_pv)  # PV-|PC
            Q += e_q * e_w_IE * np.outer(y_pv, y_pc)  # PC->PV
            U += e_u * e_w_EE * np.outer(y_pc, y_pc)  # PC->PC

            # enforce Dale's law
            W[W < 0] = 0  # L4->PC
            K[K < 0] = 0  # L4->PV
            P[P < 0] = 0  # PV-|PV
            M[M < 0] = 0  # PV-|PC
            Q[Q < 0] = 0  # PC->PV
            U[U < 0] = 0  # PC->PV

            # weight normalization
            norm_W = (np.sum(W ** l_norm, axis=1)) ** (1 / l_norm)  # EF
            norm_K = (np.sum(K ** l_norm, axis=1)) ** (1 / l_norm)  # IF
            norm_U = (np.sum(U ** l_norm, axis=1)) ** (1 / l_norm)  # EE
            norm_M = (np.sum(M ** l_norm, axis=1)) ** (1 / l_norm)  # EI
            norm_Q = (np.sum(Q ** l_norm, axis=1)) ** (1 / l_norm)  # IE
            norm_P = (np.sum(P ** l_norm, axis=1)) ** (1 / l_norm)  # II

            norm_WU = (norm_W ** l_norm + norm_U ** l_norm) ** (1 / l_norm)  # EE
            norm_KQ = (norm_K ** l_norm + norm_Q ** l_norm) ** (1 / l_norm)  # IE

            # normalization of exciatory PC input
            W *= (W_EE_norms[:, np.newaxis] / norm_WU[:, np.newaxis]) ** e_w
            U *= (W_EE_norms[:, np.newaxis] / norm_WU[:, np.newaxis]) ** e_u

            # normalization of inhibitory PC input
            M[norm_M != 0, :] *= (W_EI_norms[norm_M != 0, np.newaxis] / norm_M[
                norm_M != 0, np.newaxis]) ** e_m  # do not scale 0-norm weight vectors

            # normalization of exciatory PV input
            K *= (W_IE_norms[:, np.newaxis] / norm_KQ[:, np.newaxis]) ** e_k
            Q *= (W_IE_norms[:, np.newaxis] / norm_KQ[:, np.newaxis]) ** e_q

            # normalization of inhibitory PV input
            P[norm_P != 0, :] *= (W_II_norms[norm_P != 0, np.newaxis] / norm_P[
                norm_P != 0, np.newaxis]) ** e_p  # do not scale 0-norm weight vectors

            ##############################################################################################################################################################
            ##############################################################################################################################################################
            # BOOKKEEPING

            # record variables of interest
            if t % accuracy == 0:
                ind = int(t / accuracy)
                t_hist[ind] = t

                y_mean_L4_hist[ind] = y_mean_L4

                x_pc_hist[ind] = x_pc
                x_e_pc_hist[ind] = x_e_pc
                x_i_pc_hist[ind] = x_i_pc
                x_e_mean_pc_hist[ind] = x_e_mean_pc
                x_i_mean_pc_hist[ind] = x_i_mean_pc
                b_pc_hist[ind] = b_pc
                a_pc_hist[ind] = a_pc
                n_pc_hist[ind] = n_pc
                W_EE_norms_hist[ind] = W_EE_norms
                y_pc_hist[ind] = y_pc
                y_mean_pc_hist[ind] = y_mean_pc
                y_var_pc_hist[ind] = y_var_pc
                y_0_pc_hist[ind] = y_0_pc

                x_pv_hist[ind] = x_pv
                x_e_pv_hist[ind] = x_e_pv
                x_i_pv_hist[ind] = x_i_pv
                x_e_mean_pv_hist[ind] = x_e_mean_pv
                x_i_mean_pv_hist[ind] = x_i_mean_pv
                b_pv_hist[ind] = b_pv
                a_pv_hist[ind] = a_pv
                n_pv_hist[ind] = n_pv
                W_IE_norms_hist[ind] = W_IE_norms
                y_pv_hist[ind] = y_pv
                y_mean_pv_hist[ind] = y_mean_pv
                y_var_pv_hist[ind] = y_var_pv
                y_0_pv_hist[ind] = y_0_pv

                # record weight data from one neuron
                W_hist[ind] = W[0]
                W_norm_hist[ind] = np.sum(W, axis=1)[0]
                K_hist[ind] = K[0]
                K_norm_hist[ind] = np.sum(K, axis=1)[0]
                P_hist[ind] = P[0]
                P_norm_hist[ind] = np.sum(P, axis=1)[0]
                M_hist[ind] = M[0]
                M_norm_hist[ind] = np.sum(M, axis=1)[0]
                Q_hist[ind] = Q[0]
                Q_norm_hist[ind] = np.sum(Q, axis=1)[0]
                U_hist[ind] = U[0]
                U_norm_hist[ind] = np.sum(U, axis=1)[0]

            # during last 'T_fine' iterations, record variables at every iteration
            if t >= T - T_fine:
                ind = t - (T - T_fine)

                x_pc_hist_fine[ind] = x_pc
                x_e_pc_hist_fine[ind] = x_e_pc
                x_i_pc_hist_fine[ind] = x_i_pc
                x_e_mean_pc_hist_fine[ind] = x_e_mean_pc
                x_i_mean_pc_hist_fine[ind] = x_i_mean_pc
                b_pc_hist_fine[ind] = b_pc
                a_pc_hist_fine[ind] = a_pc
                n_pc_hist_fine[ind] = n_pc
                W_EE_norms_hist_fine[ind] = W_EE_norms
                y_pc_hist_fine[ind] = y_pc
                y_mean_pc_hist_fine[ind] = y_mean_pc  # average pyramidal cell hist
                y_var_pc_hist_fine[ind] = y_var_pc
                y_0_pc_hist_fine[ind] = y_0_pc

                x_pv_hist_fine[ind] = x_pv
                x_e_pv_hist_fine[ind] = x_e_pv
                x_i_pv_hist_fine[ind] = x_i_pv
                x_e_mean_pv_hist_fine[ind] = x_e_mean_pv
                x_i_mean_pv_hist_fine[ind] = x_i_mean_pv
                b_pv_hist_fine[ind] = b_pv
                a_pv_hist_fine[ind] = a_pv
                n_pv_hist_fine[ind] = n_pv
                W_IE_norms_hist_fine[ind] = W_IE_norms
                y_pv_hist_fine[ind] = y_pv
                y_mean_pv_hist_fine[ind] = y_mean_pv  # interneuron history
                y_var_pv_hist_fine[ind] = y_var_pv
                y_0_pv_hist_fine[ind] = y_0_pv

                y_L4_hist_fine[ind] = y_L4
                y_mean_L4_hist_fine[ind] = y_mean_L4  # mean activity of layer 4 inputs

                W_hist_fine[ind, :, :] = W
                K_hist_fine[ind, :, :] = K
                P_hist_fine[ind, :, :] = P
                M_hist_fine[ind, :, :] = M
                Q_hist_fine[ind, :, :] = Q
                U_hist_fine[ind, :, :] = U

            if t % accuracy_states == 0:
                ind = int(t / accuracy_states) + 1
                network_states[ind] = copy.deepcopy([t] + network)

            # save data every 10% of runtime
            if ((t + 1) % int(T / 10) == 0) * online_saves:
                io.save([network_states, hist, param], save_path_network + "/data.p")
                print("network saved")

    print("-----------------")

    network_states[-1] = copy.deepcopy([t] + network)
    io.save([network_states, hist, param], save_path_network + "/data.p")
    print("network saved")

    ##############################################################################################################################################################
    ##############################################################################################################################################################
    # Epilogue

    if run_analysis:
        print('analyse tuning and connectivity')
        os.system("python3 analyze_tuning_and_connectivity.py")

    print("-----------------")

    ###############################################################################
    # copy latest simulation figures to '_current' folder
    src = save_path_network + "/"
    dirc = "data/_current/"

    # create folder structure in 'data/_current' and delete existing content
    for dirpath, dirnames, filenames in os.walk(src):
        structure = os.path.join(dirc, dirpath[len(src):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        for ele in filenames:
            if os.path.exists(os.path.join(structure, ele)):
                os.unlink(os.path.join(structure, ele))

    # copy files
    copy_tree(src, dirc)

    ###############################################################################
    # plot runtime in console

    runtime = time.time() - start_time
    print("runtime: ", end='', flush=True)
    print(int(np.floor(runtime / 60)), end='')
    print("min ", end='')
    print(int(runtime - np.floor(runtime / 60) * 60), end='')
    print("sec")

    print("-----------------")

if __name__ == '__main__':
    main()