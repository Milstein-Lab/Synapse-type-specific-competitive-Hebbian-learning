
##############################################################################################################################################################
##############################################################################################################################################################
# IMPORTS

import numpy as np
from numpy.random import RandomState
import scipy as sp
from scipy import ndimage
from scipy.optimize import curve_fit
from random import shuffle
import matplotlib
import matplotlib.pyplot as plt; plt.close('all');
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import h5py
import copy
import timeit
import time
import datetime
import sys
import os
import shutil
from distutils.dir_util import copy_tree

import importlib.util

# import lib.inputs as inputs  #should be loaded from specific simulation (see below)
import libraries.loadsave as io
import libraries.plots as plots
import libraries.functions as fn

np.seterr(all='raise')  # raise error in case of RuntimeWarning

##############################################################################################################################################################
##############################################################################################################################################################
# LOAD NETWORK

# load most recent simulation
dirs = os.listdir("data/")
dates = np.char.ljust(dirs, 21) # get data string of subfolderds in 'data/' directory
dates = np.char.replace(dates,'-', '')  # remove '-' separator
dates[np.logical_not(np.char.isnumeric(dates))] = '0'  # set all non-numeric values to '0' to enable conversion
load_path_network = "data/"+dirs[np.argmax(dates.astype(float))]  # convert dates + sim start times to number and pick dir with max number

# or set load path manually
# load_path_network = '/folder'

save_path_network = load_path_network  # save results to original sim data folder

# load lib/inputs from original simulation
spec = importlib.util.spec_from_file_location("inputs", load_path_network+"/libraries/inputs.py")
inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inputs)

# load network
loaded_data = io.load(load_path_network+"/data.p")

save_path_figures = save_path_network+"/figures"
fn.create_directory(save_path_figures)

shutil.copy2(os.path.basename(__file__), save_path_network+"/"+os.path.basename(__file__))

#######################################################################################################################

# load original network at end of original sim
[t,
 W, K, P, M, Q, U,
 a_pc, b_pc, n_pc, W_EF_norms, W_EI_norms, W_EE_norms, y_mean_pc, y_var_pc, y_0_pc,
 a_pv, b_pv, n_pv, W_IF_norms, W_II_norms, W_IE_norms, y_mean_pv, y_var_pv, y_0_pv] = loaded_data[0][-1]

n_pc = 80
n_pv = 20
W_EF_norms = W_EE_norms
W_IF_norms = W_IE_norms

it = iter(np.arange(len(loaded_data[0][-1])-1)+1) # create iterator
net_dic = {"W":next(it), "K":next(it), "P":next(it), "M":next(it), "Q":next(it), "U":next(it),
           "a\_pc":next(it), "b\_pc":next(it), "d\_pc":next(it), "h\_pc":next(it), "y\_mean\_pc":next(it), "y\_var\_pc":next(it), "y\_0\_pc":next(it),
           "a\_pv":next(it), "b\_pv":next(it), "d\_pv":next(it), "h\_pv":next(it), "y\_mean\_pv":next(it), "y\_var\_pv":next(it), "y\_0\_pv":next(it)}

# load original hist
[x_pc_hist, x_e_pc_hist, x_i_pc_hist, x_e_mean_pc_hist, x_i_mean_pc_hist, b_pc_hist, a_pc_hist, n_pc_hist, W_EF_norms_hist, W_EI_norms_hist, W_EE_norms_hist, y_pc_hist, y_mean_pc_hist, y_var_pc_hist, y_0_pc_hist,
 x_pv_hist, x_e_pv_hist, x_i_pv_hist, x_e_mean_pv_hist, x_i_mean_pv_hist, b_pv_hist, a_pv_hist, n_pv_hist, W_IF_norms_hist, W_II_norms_hist, W_IE_norms_hist, y_pv_hist, y_mean_pv_hist, y_var_pv_hist, y_0_pv_hist,
 W_hist, K_hist, P_hist, M_hist, Q_hist, U_hist,
 W_norm_hist, K_norm_hist, P_norm_hist, M_norm_hist, Q_norm_hist, U_norm_hist] = loaded_data[1]

#load original network parameters
[e_x_pc, e_x_pv, e_y, e_w_EE, e_w_IE, e_w_EI, e_w_II,
 N_L4, N_pc, N_pv, input_amp, input_sigma] = loaded_data[2]

T_seq = 100  # how many iterations per stimulus
n_probe = 100  # how many different stimuli to probe
T = n_probe*T_seq

#######################################################################################################################
# Define some variables

# L4 input
y_L4 = np.zeros((N_L4,))  # L4 input
y_mean_L4 = np.zeros((N_L4,))

# L2/3 PYR principal cells (pc)
y_pc = np.zeros((N_pc,))
x_pc = np.zeros((N_pc,))
x_e_pc = np.zeros((N_pc,))
x_i_pc = np.zeros((N_pc,))
x_e_mean_pc = np.zeros((N_pc,))
x_i_mean_pc = np.zeros((N_pc,))

# L2/3 PV inhibitory cells (pv)
y_pv = np.zeros((N_pv,))
x_pv = np.zeros((N_pv,))
x_e_pv = np.zeros((N_pv,))
x_i_pv = np.zeros((N_pv,))
x_e_mean_pv = np.zeros((N_pv,))
x_i_mean_pv = np.zeros((N_pv,))

###############################################################################
# history preallocation

t_hist_fine = np.zeros((T,))
theta_hist_fine = np.zeros((T,))

# PC
x_pc_hist_fine        = np.zeros((T, N_pc))
x_e_pc_hist_fine      = np.zeros((T, N_pc))
x_i_pc_hist_fine      = np.zeros((T, N_pc))
x_e_mean_pc_hist_fine = np.zeros((T, N_pc))
x_i_mean_pc_hist_fine = np.zeros((T, N_pc))
y_pc_hist_fine        = np.zeros((T, N_pc))
y_mean_pc_hist_fine   = np.zeros((T, N_pc))
y_var_pc_hist_fine    = np.zeros((T, N_pc))

# PV
x_pv_hist_fine        = np.zeros((T, N_pv))
x_e_pv_hist_fine      = np.zeros((T, N_pv))
x_i_pv_hist_fine      = np.zeros((T, N_pv))
x_e_mean_pv_hist_fine = np.zeros((T, N_pv))
x_i_mean_pv_hist_fine = np.zeros((T, N_pv))
y_pv_hist_fine        = np.zeros((T, N_pv))
y_mean_pv_hist_fine   = np.zeros((T, N_pv))
y_var_pv_hist_fine    = np.zeros((T, N_pv))

# L4
y_L4_hist_fine      = np.zeros((T, N_L4))
y_mean_L4_hist_fine = np.zeros((T, N_L4))

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#------------ SIM ------------------


# for t in tqdm(range(T)):
for t in range(T):

    # load input patch
    if t % T_seq == 0:  # switch input every 'T_seq' iterations
        orientation = ( (t / T_seq) % n_probe ) / n_probe * np.pi  # sweep over all orientations. Probe 'n_probe' orientations, 'T_seq' iterations per orientation
        y_L4 = inputs.get_input(N_L4, theta=orientation, sigma=input_sigma*np.pi/180, amp=input_amp)

        # reset network activity
        y_pc *= 0
        y_pv *= 0
        x_e_pc *= 0
        x_i_pc *= 0
        x_e_pv *= 0
        x_i_pv *= 0

    x_e_pc += e_x_pc * (-x_e_pc + np.dot(W, y_L4) + np.dot(U, y_pc))
    x_i_pc += e_x_pc * (-x_i_pc + np.dot(M, y_pv))
    x_pc = x_e_pc - x_i_pc

    x_e_pv += e_x_pv * (-x_e_pv + np.dot(K, y_L4) + np.dot(Q, y_pc))
    x_i_pv += e_x_pv * (-x_i_pv + np.dot(P, y_pv))
    x_pv = x_e_pv - x_i_pv

    x_pc = x_pc.clip(min=0)  # clip negative membrane potential values to zero
    x_pv = x_pv.clip(min=0)

    y_pc = a_pc*(x_pc**2)   # get firing rates
    y_pv = a_pv*(x_pv**2)  # get firing rates

###############################################################################
    # running averages + variances
    y_mean_pc_old = y_mean_pc
    y_mean_pc += e_y * (-y_mean_pc + y_pc)
    y_var_pc += e_y * (-y_var_pc + (y_pc-y_mean_pc_old)*(y_pc-y_mean_pc))
    x_e_mean_pc += e_y * (-x_e_mean_pc + x_e_pc)
    x_i_mean_pc += e_y * (-x_i_mean_pc + x_i_pc)

    y_mean_pv_old = y_mean_pv
    y_mean_pv += e_y * (-y_mean_pv + y_pv)
    y_var_pv += e_y * (-y_var_pv + (y_pv-y_mean_pv_old)*(y_pv-y_mean_pv))
    x_e_mean_pv += e_y * (-x_e_mean_pv + x_e_pv)
    x_i_mean_pv += e_y * (-x_i_mean_pv + x_i_pv)

###############################################################################
# record variables in every iterations

    t_hist_fine[t]           = t
    theta_hist_fine[t]       = orientation

    x_pc_hist_fine[t]        = x_pc
    x_e_pc_hist_fine[t]      = x_e_pc
    x_i_pc_hist_fine[t]      = x_i_pc
    x_e_mean_pc_hist_fine[t] = x_e_mean_pc
    x_i_mean_pc_hist_fine[t] = x_i_mean_pc
    y_pc_hist_fine[t]        = y_pc
    y_mean_pc_hist_fine[t]   = y_mean_pc
    y_var_pc_hist_fine[t]    = y_var_pc

    x_pv_hist_fine[t]        = x_pv
    x_e_pv_hist_fine[t]      = x_e_pv
    x_i_pv_hist_fine[t]      = x_i_pv
    x_e_mean_pv_hist_fine[t] = x_e_mean_pv
    x_i_mean_pv_hist_fine[t] = x_i_mean_pv
    y_pv_hist_fine[t]        = y_pv
    y_mean_pv_hist_fine[t]   = y_mean_pv
    y_var_pv_hist_fine[t]    = y_var_pv

    y_L4_hist_fine[t]        = y_L4
    y_mean_L4_hist_fine[t]   = y_mean_L4

print("sim ready")

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#------------ PLOTS ------------------
if plot:

    # calculate tuning curves + E/I balance of all PV and PC cells
    #-------------------------------------

    # take average activity for each orientation (averaged over second half of trial T_seq to mitigate transient dynamics)
    x_e_pc_stim_avg = np.mean(np.reshape(x_e_pc_hist_fine, (int(T/T_seq), T_seq, N_pc) )[:,int(T_seq*2/4):,:], axis=1)
    x_i_pc_stim_avg = np.mean(np.reshape(x_i_pc_hist_fine, (int(T/T_seq), T_seq, N_pc) )[:,int(T_seq*2/4):,:], axis=1)
    y_pc_stim_avg   = np.mean(np.reshape(y_pc_hist_fine  , (int(T/T_seq), T_seq, N_pc) )[:,int(T_seq*2/4):,:], axis=1)

    x_e_pv_stim_avg = np.mean(np.reshape(x_e_pv_hist_fine, (int(T/T_seq), T_seq, N_pv) )[:,int(T_seq*2/4):,:], axis=1)
    x_i_pv_stim_avg = np.mean(np.reshape(x_i_pv_hist_fine, (int(T/T_seq), T_seq, N_pv) )[:,int(T_seq*2/4):,:], axis=1)
    y_pv_stim_avg   = np.mean(np.reshape(y_pv_hist_fine  , (int(T/T_seq), T_seq, N_pv) )[:,int(T_seq*2/4):,:], axis=1)

    y_L4_stim_avg = np.mean(np.reshape(y_L4_hist_fine  , (int(T/T_seq), T_seq, N_L4) )[:,int(T_seq*2/4):,:], axis=1)

    y_limit = 1.1 * np.max([np.max(elem) for elem in [x_e_pc_stim_avg, x_i_pc_stim_avg, x_e_pv_stim_avg, x_i_pv_stim_avg] ])  # y_axis limit for plotting

    theta_stim_avg = theta_hist_fine[0::T_seq]

    # rearrange neurons according to preferred tunings
    sorted_pc_ids = np.argsort(np.argmax(y_pc_stim_avg, axis=0))
    sorted_pv_ids = np.argsort(np.argmax(y_pv_stim_avg, axis=0))
    sorted_L4_ids = np.argsort(np.argmax(y_L4_stim_avg, axis=0))

    # save tuning curve peak of pc and pc neuron_class
    #-------------------------------------
    pc_tuning_peak = theta_stim_avg[np.argmax(y_pc_stim_avg, axis=0)]
    pv_tuning_peak = theta_stim_avg[np.argmax(y_pv_stim_avg, axis=0)]
    L4_tuning_peak = theta_stim_avg[np.argmax(y_L4_stim_avg, axis=0)]

    io.save([pc_tuning_peak, pv_tuning_peak], save_path_network+"/tuning_peaks.p")
    print("tuning peaks saved")


    # plot average tuning curves
    #-------------------------------------

    # PC
    # add periods before cutting around tuning peak
    theta_stim_avg_centered = np.concatenate((theta_stim_avg-np.pi,theta_stim_avg,theta_stim_avg+np.pi), axis=0)
    y_pc_stim_avg_centered = np.concatenate((y_pc_stim_avg,y_pc_stim_avg,y_pc_stim_avg), axis =0)
    x_e_pc_stim_avg_centered = np.concatenate((x_e_pc_stim_avg,x_e_pc_stim_avg,x_e_pc_stim_avg), axis =0)
    x_i_pc_stim_avg_centered = np.concatenate((x_i_pc_stim_avg,x_i_pc_stim_avg,x_i_pc_stim_avg), axis =0)

    # gather indices for cutting around tuning peak for each neuron
    start = (n_probe+np.argmax(y_pc_stim_avg, axis=0)-n_probe/2).astype(int)
    stop = (n_probe/2+np.argmax(y_pc_stim_avg, axis=0)+n_probe).astype(int)
    idx = np.array([ [i for i in np.arange(start[j], stop[j])] for j in np.arange(start.shape[0])])

    # cut region around tuning peak
    theta_delta_pc = theta_stim_avg_centered[idx] - pc_tuning_peak[:,np.newaxis]
    y_pc_stim_avg_centered =  y_pc_stim_avg_centered[idx.T,np.arange(N_pc)]
    x_e_pc_stim_avg_centered = x_e_pc_stim_avg_centered[idx.T,np.arange(N_pc)]
    x_i_pc_stim_avg_centered = x_i_pc_stim_avg_centered[idx.T,np.arange(N_pc)]

    # PV
    # add periods before cutting around tuning peak
    y_pv_stim_avg_centered = np.concatenate((y_pv_stim_avg,y_pv_stim_avg,y_pv_stim_avg), axis =0)  # add periods before cutting around tuning peak
    x_e_pv_stim_avg_centered = np.concatenate((x_e_pv_stim_avg,x_e_pv_stim_avg,x_e_pv_stim_avg), axis =0)
    x_i_pv_stim_avg_centered = np.concatenate((x_i_pv_stim_avg,x_i_pv_stim_avg,x_i_pv_stim_avg), axis =0)

    # gather indices for cutting around tuning peak for each neuron
    start = (n_probe+np.argmax(y_pv_stim_avg, axis=0)-n_probe/2).astype(int)
    stop = (n_probe/2+np.argmax(y_pv_stim_avg, axis=0)+n_probe).astype(int)
    idx = np.array([ [i for i in np.arange(start[j], stop[j])] for j in np.arange(start.shape[0])])

    # cut region around tuning peak
    theta_delta_pv = theta_stim_avg_centered[idx] - pv_tuning_peak[:,np.newaxis]
    y_pv_stim_avg_centered =  y_pv_stim_avg_centered[idx.T,np.arange(N_pv)]
    x_e_pv_stim_avg_centered = x_e_pv_stim_avg_centered[idx.T,np.arange(N_pv)]
    x_i_pv_stim_avg_centered = x_i_pv_stim_avg_centered[idx.T,np.arange(N_pv)]


    # plot average E and I input to an E neuron
    #-------------------------------------
    x_data = [np.mean(theta_delta_pc,axis=0)*180/np.pi, np.mean(theta_delta_pc,axis=0)*180/np.pi]
    y_data = [np.mean(x_e_pc_stim_avg_centered, axis=1), np.mean(x_i_pc_stim_avg_centered, axis=1)]
    y_data = [y_data[0]/np.max(y_data[0]), y_data[1]/np.max(y_data[1])]  # normalize
    plots.line_plot(
        x_data,
        y_data,
        colors=['blue', 'red'],
        linestyles=['--','--'],
        labels=[r'$x_E$', r'$x_I$'],
        yticks = [0, 1],
        y_max=1,
        y_min=0,
        ytickslabels=['0','1'],
        xticks=[-90,-45,0,45,90],
        xticklabels= ['',r'$-45^{\circ}$',r'$0^{\circ}$',r'$45^{\circ}$',''],
        xlabel=r'$\Delta \theta$',
        ylabel=r'Input to E (norm.)',
        figsize=[2.5,2.0],
        save_path=save_path_figures+'/average_E_I_tuning_pc.pdf')


    # plot average E and I input to an I neuron
    #-------------------------------------
    x_data = [np.mean(theta_delta_pv,axis=0)*180/np.pi, np.mean(theta_delta_pv,axis=0)*180/np.pi]
    y_data = [np.mean(x_e_pv_stim_avg_centered, axis=1), np.mean(x_i_pv_stim_avg_centered, axis=1)]
    y_data = [y_data[0]/np.max(y_data[0]), y_data[1]/np.max(y_data[1])]  # normalize
    plots.line_plot(
        x_data,
        y_data,
        colors=['blue', 'red'],
        linestyles=['--','--'],
        labels=[r'$x_E$', r'$x_I$'],
        # yticks=[0,0.25,0.5,0.75,1],
        yticks = [0, 1],
        # yticklabels= ['0','','','','1'],
        y_max=1,
        y_min=0,
        ytickslabels=['0','1'],
        xticks=[-90,-45,0,45,90],
        xticklabels= ['',r'$-45^{\circ}$',r'$0^{\circ}$',r'$45^{\circ}$',''],
        xlabel=r'$\Delta \theta$',
        ylabel=r'Input to I (norm.)',
        figsize=[2.5,2.0],
        save_path=save_path_figures+'/average_E_I_tuning_pv.pdf')


    # plot average tuning curves - compare PV,PC firing rate tuning
    #-------------------------------------
    x_data = [np.mean(theta_delta_pc,axis=0)*180/np.pi, np.mean(theta_delta_pv,axis=0)*180/np.pi]
    y_data = [np.mean(y_pc_stim_avg_centered, axis=1), np.mean(y_pv_stim_avg_centered, axis=1)]
    y_data = [y_data[0]/np.max(y_data[0]), y_data[1]/np.max(y_data[1])]  # normalize
    plots.line_plot(
        x_data,
        y_data,
        colors=['blue', 'red', 'blue', 'red'],
        linestyles=['-','-','--','--'],
        labels=[r'$y_E$', r'$y_I$'],
        # yticks=[0,0.25,0.5,0.75,1],
        yticks = [0, 1],
        # yticklabels= ['0','','','','1'],
        y_max=1,
        y_min=0,
        ytickslabels=['0','1'],
        xticks=[-90,-45,0,45,90],
        xticklabels= ['',r'$-45^{\circ}$',r'$0^{\circ}$',r'$45^{\circ}$',''],
        xlabel=r'$\Delta \theta$',
        ylabel=r'Firing rate (norm.)',
        figsize=[2.5,2.0],
        save_path=save_path_figures+'/average_firing_rate_tuning_pc_pv.pdf')

    idx = sorted_pc_ids[int(N_pc/2)]  # get neuron that is closest tuned to 90 degrees
    data = [x_i_pc_stim_avg[:,idx]]
    linestyles = ['-']
    colors = ['darkred']
    alphas = [1]
    for i in range(N_pv):
        data.append(M[idx,i] * y_pv_stim_avg[:,i])
        linestyles.append('-')
        colors.append('red')
        alphas.append(0.6)

    plots.line_plot(
        [theta_stim_avg*180/np.pi],
        data[1:],
        random_colors = [['darkred', 'red', 'lightred']],
        yticks = [],
        ytickslabels=[],
        xticks=[0,45,90,135,180],
        xticklabels= ['',r'$45^{\circ}$',r'',r'$135^{\circ}$',''],
        spine_visibility = [1,0,0,0],
        xlabel=r'$\theta$',
        ylabel=r'Inhib. input',
        figsize=[2.5,2.0],
        save_path=save_path_figures+'/inhibitory_input_composition_pc.pdf')


    # plot connectivity matrices (sorted PV/PC)
    #-------------------------------------
    U_selector = tuple(np.meshgrid(sorted_pc_ids,sorted_pc_ids, indexing='ij'))
    Q_selector = tuple(np.meshgrid(sorted_pv_ids,sorted_pc_ids, indexing='ij'))
    P_selector = tuple(np.meshgrid(sorted_pv_ids,sorted_pv_ids, indexing='ij'))
    M_selector = tuple(np.meshgrid(sorted_pc_ids,sorted_pv_ids, indexing='ij'))

    U_sorted = U[U_selector]
    Q_sorted = Q[Q_selector]
    P_sorted = P[P_selector]
    M_sorted = M[M_selector]

    vmax = np.max([np.max(elem) for elem in [U, Q, P, M] ])

    plots.connectivity_matrix([U_sorted, M_sorted, Q_sorted, P_sorted], xlabel=r'pre', ylabel=r'post', save_path=save_path_figures+'/recurrent_connectivity_matrix.pdf')


    # plot connectivity kernels as function of tuning difference between pre- and postsynaptic neurons
    #-------------------------------------
    pc_tuning_peak *= 180/np.pi
    pv_tuning_peak *= 180/np.pi
    L4_tuning_peak *= 180/np.pi

    #PC
    delta_theta_W = L4_tuning_peak-pc_tuning_peak[:, np.newaxis]
    delta_theta_W = ((delta_theta_W+3*90)%180)-90  # correct for cyclic distance
    delta_theta_U = pc_tuning_peak-pc_tuning_peak[:, np.newaxis]
    delta_theta_U = ((delta_theta_U+3*90)%180)-90  # correct for cyclic distance
    delta_theta_M = pv_tuning_peak-pc_tuning_peak[:, np.newaxis]
    delta_theta_M = ((delta_theta_M+3*90)%180)-90  # correct for cyclic distance
    plots.scatter_plot(
        [delta_theta_W.flatten(), delta_theta_U.flatten(), delta_theta_M.flatten()],
        [W.flatten(), U.flatten(), M.flatten()],
        fit_gaussian=True,
        size = 1,
        colors=['lightblue', 'blue', 'red'],
        labels=[r'$W_{EF}$', r'$W_{EE}$', r'$W_{EI}$'],
        xticks=[-90,-45,0,45,90],
        xticklabels= ['',r'$-45^{\circ}$',r'$0^{\circ}$',r'$45^{\circ}$',''],
        xlabel=r'$\Delta \theta$',
        ylabel=r'syn. weight',
        legend=True,
        save_path=save_path_figures+'/connectivity_kernels_pc.pdf')

    #PV
    delta_theta_K = L4_tuning_peak-pv_tuning_peak[:, np.newaxis]
    delta_theta_K = ((delta_theta_K+3*90)%180)-90  # correct for cyclic distance
    delta_theta_Q = pc_tuning_peak-pv_tuning_peak[:, np.newaxis]
    delta_theta_Q = ((delta_theta_Q+3*90)%180)-90  # correct for cyclic distance
    delta_theta_P = pv_tuning_peak-pv_tuning_peak[:, np.newaxis]
    delta_theta_P = ((delta_theta_P+3*90)%180)-90  # correct for cyclic distance
    plots.scatter_plot(
        [delta_theta_K.flatten(), delta_theta_Q.flatten(), delta_theta_P.flatten()],
        [K.flatten(), Q.flatten(), P.flatten()],
        fit_gaussian=True,
        size = 1,
        colors=['lightblue', 'blue', 'red'],
        labels=[r'$W_{IF}$', r'$W_{IE}$', r'$W_{II}$'],
        xticks=[-90,-45,0,45,90],
        xticklabels= ['',r'$-45^{\circ}$',r'$0^{\circ}$',r'$45^{\circ}$',''],
        xlabel=r'$\Delta \theta$',
        ylabel=r'syn. weight',
        legend=True,
        save_path=save_path_figures+'/connectivity_kernels_pv.pdf')

    #   plot recurrent kernels
    #-------------------------------------
    def gaussian(x, amp, mean, sigma):
        return amp*np.exp(-(x - mean)**2/(2*sigma**2))

    x_data = [delta_theta_U.flatten(), delta_theta_M.flatten(), delta_theta_Q.flatten(), delta_theta_P.flatten()]
    y_data = [U.flatten(), -M.flatten(), Q.flatten(), -P.flatten()]
    y_data_fit = [[]] * len(y_data)
    for i in range(len(x_data)):
        amp_0 = np.sign(np.mean(y_data[i])) *np.max(abs(y_data[i]))
        if np.max(y_data[i]) == np.mean(y_data[i]):
            y_data_fit[i] = np.zeros((100,))  # if max equals min, everything is zero
        else:
            x_min = np.min(x_data[i])
            x_max = np.max(x_data[i])
            mean_0 = 0  #np.mean(x_data[i])
            std_0 = np.std(x_data[i])/2
            popt, pcov = curve_fit(f=gaussian, xdata=x_data[i], ydata=y_data[i], p0=[amp_0, mean_0, std_0], bounds=((-abs(amp_0), x_min, (x_max-x_min)/100), (abs(amp_0), x_max, (x_max-x_min))) )
            x_samples = np.linspace(x_min, x_max, num = 100)
            y_data_fit[i] = gaussian(x_samples, *popt)
            y_data_fit[i] /= np.max(abs(y_data_fit[i])) # normalize to 1

    plots.line_plot(
        [x_samples],
        y_data_fit,
        colors=['blue', 'red', 'darkblue', 'darkred'],
        linestyles=['-','-',(0,(2,3)),(0,(2,3))],
        labels=[r'$W_{EE}$', r'$W_{EI}$', r'$W_{IE}$', r'$W_{II}$'],
        xticks=[-90,-45,0,45,90],
        xticklabels= ['',r'$-45^{\circ}$',r'$0^{\circ}$',r'$45^{\circ}$',''],
        yticks=[-1,0,1,2],
        yticklabels= ['-1','0','1',''],
        y_max=1.5,
        xlabel=r'$\Delta \theta$',
        ylabel=r'Syn. weight (norm.)',
        figsize=[2.5,2.0],
        save_path=save_path_figures+'/connectivity_kernels_recurrent.pdf')


    # plot feed forward weights before, during, and after training
    #-------------------------------------
    W_hist = fn.extract_weight_hist(loaded_data[0] ,net_dic['W'], downSampleRatio=1)
    K_hist = fn.extract_weight_hist(loaded_data[0] ,net_dic['K'], downSampleRatio=1)

    filetype = 'pdf'


    # PV ffwd weights
    # Plot K at end of simulation
    K = K_hist[-1,:,:]
    K = np.hstack((K,K[:,[0]]))
    plots.line_plot([np.linspace(0,180,K.shape[1])], [K.T], figsize=[4.2,1.8], random_colors=[['darkred','red', 'lightred']], spine_visibility=[1,0,0,0], yticks=[], yticklabels=[], xticks=[0, 45, 90, 135, 180], xticklabels=['','$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='', save_path=save_path_figures+'/K_after.'+filetype)

    # Plot K at beginning of simulation
    K = K_hist[5,:,:]
    K = np.hstack((K,K[:,[0]]))
    plots.line_plot([np.linspace(0,180,K.shape[1])], [K.T], figsize=[4.2,1.8], random_colors=[['darkred','red', 'lightred']], spine_visibility=[1,0,0,0], yticks=[], yticklabels=[], xticks=[0, 45, 90, 135, 180], xticklabels=['','$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='', save_path=save_path_figures+'/K_before.'+filetype)

    # Plot K at initialization of simulation
    K = K_hist[0,:,:]
    K = np.hstack((K,K[:,[0]]))
    plots.line_plot([np.linspace(0,180,K.shape[1])], [K.T], figsize=[4.2,1.8], random_colors=[['darkred','red', 'lightred']], spine_visibility=[1,0,0,0], yticks=[], yticklabels=[], xticks=[0, 45, 90, 135, 180], xticklabels=['','$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='', save_path=save_path_figures+'/K_init.'+filetype)


    # PC ffwd weights
    # Plot W at end of simulation
    W = W_hist[-1,:,:]
    W = np.hstack((W,W[:,[0]]))
    plots.line_plot([np.linspace(0,180,W.shape[1])], [W.T], figsize=[4.2,1.8], random_colors=[['darkblue','blue', 'lightblue']], spine_visibility=[1,0,0,0], yticks=[], yticklabels=[], xticks=[0, 45, 90, 135, 180], xticklabels=['','$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='', save_path=save_path_figures+'/W_after.'+filetype)

    # Plot W at beginning of simulation
    W = W_hist[5,:,:]
    W = np.hstack((W,W[:,[0]]))
    plots.line_plot([np.linspace(0,180,W.shape[1])], [W.T], figsize=[4.2,1.8], random_colors=[['darkblue','blue', 'lightblue']], spine_visibility=[1,0,0,0], yticks=[], yticklabels=[], xticks=[0, 45, 90, 135, 180], xticklabels=['','$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='', save_path=save_path_figures+'/W_before.'+filetype)

    # Plot W at initialization of simulation
    W = W_hist[0,:,:]
    W = np.hstack((W,W[:,[0]]))
    plots.line_plot([np.linspace(0,180,W.shape[1])], [W.T], figsize=[4.2,1.8], random_colors=[['darkblue','blue', 'lightblue']], spine_visibility=[1,0,0,0], yticks=[], yticklabels=[], xticks=[0, 45, 90, 135, 180], xticklabels=['','$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='', save_path=save_path_figures+'/W_init.'+filetype)
