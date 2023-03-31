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
import matplotlib.pyplot as plt;
plt.close('all');
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
import click


@click.command()
@click.option('--config_file_path', type=click.Path(exists=True), required=False, default='config/orig_config.yaml')
@click.option('--data_file_path', type=click.Path(exists=True), required=False)
@click.option('--plot', is_flag=True)
@click.option('--run_analysis', is_flag=True)
@click.option('--export', is_flag=True)
@click.option('--debug', type=bool, default=False)
@click.option('--run_sim', type=bool, default=True)
def main(config_file_path, data_file_path, plot, run_analysis, export, debug, run_sim):


    import init
    importlib.reload(init)  # make sure to load most recent init version
    #from init import *  # import config file

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