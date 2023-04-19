import numpy as np
from numpy.random import RandomState
import scipy as sp
from scipy import optimize
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
import libraries.functions as fn
import libraries.plots as plots
import init
import click


class Network(object):
    def __init__(self,short_description,random_seed,online_saves,run_analysis,T_train,T_seq_train,T_fine_train,N_pc,N_pv,N_L4,gain,bias,n_exp,
                 W_EE_norm,W_EI_norm,W_IE_norm,W_II_norm,W_EF_norm,W_IF_norm,e_x_pc_inv,e_x_pv_inv,e_y,std_W_EL,
                 std_W_IL,std_W_II,std_W_EI,std_W_IE,std_W_EE,tuned_init,sigma_ori,input_sigma,input_amp,l_norm,
                 joint_norm,lateral_norm,e,e_w,e_k,e_p,e_m,e_q,e_u, export=False, export_dir_path=None,
                 export_file_name=None):
        """
        This is what this class does.
        :param short_description: file name
        :param random_seed: sets random seed
        :param online_saves: If ==1, during simulation, periodically save all network weights
        :param run_analysis: analyse simulation results and produce plots (deprecated)
        :param T_train: number of simulated timesteps
        :param T_seq_train: length of one sequence - how long one orientation is presented. Corresponds to 200ms, since one iteration (Delta t) corresponds to 10ms .
        :param T_fine_train: record network at single timestep resolution during the last 'T_fine' iterations of the simulation
        :param N_pc: number of excitatory neurons
        :param N_pv: number of inhibitory neurons
        :param N_L4: number of input neurons
        :param gain: gain 'a' of activation function a(x-b)_+^n
        :param bias: bias 'b' of activation function
        :param n_exp: exponent 'n' of activation function
        :param W_EE_norm: initialze synaptic weight norms of E to E (becomes non-zero during simulation)
        :param W_EI_norm: initialze synaptic weight norms of E to I
        :param W_IE_norm: initialze synaptic weight norms of I to E (becomes non-zero during simulation)
        :param W_II_norm: initialze synaptic weight norms of I to I
        :param W_EF_norm:
        :param W_IF_norm:
        :param e_x_pc_inv: corresponds to tau_E = 1/e_x_pc * Delta_t = 20ms, where Delta_t = 10ms
        :param e_x_pv_inv: corresponds to tau_I = 1/e_x_pv * Delta_t = 17ms, where Delta_t = 10ms
        :param e_y: timescale of online exponential weighted average to track mean firing rates and variances
        :param std_W_EL: initial weights drawn from half-normal distribution with following stds (L4->PC)
        :param std_W_IL: initial weights drawn from half-normal distribution with following stds (L4->PV)
        :param std_W_II: initial weights drawn from half-normal distribution with following stds (PV-|PV)
        :param std_W_EI: initial weights drawn from half-normal distribution with following stds (PV-|PC)
        :param std_W_IE: initial weights drawn from half-normal distribution with following stds (PC->PV)
        :param std_W_EE: initial weights drawn from half-normal distribution with following stds (PC->PC)
        :param tuned_init: boolean, initialize all weights already tuned and distributed across the stimulus space (for testing purposes only)
        :param sigma_ori: stdv of preinitialized weight kernels
        :param input_sigma: width of input cell's tuning curves
        :param input_amp: maximum response of input cells at tuning peak
        :param l_norm: choose L'n'-norm (L1, L2, ...)
        :param joint_norm: normalize all excitatory and inhibitory inputs together
        :param lateral_norm: normalize all input streams separately: feedforward excitation, lateral excitation, lateral inhibition
        :param e: plasticity timescale multiplier
        :param e_w: plasticity on/off switch of specific weight matrices EF
        :param e_k: plasticity on/off switch of specific weight matrices IF
        :param e_p: plasticity on/off switch of specific weight matrices II
        :param e_m: plasticity on/off switch of specific weight matrices EI
        :param e_q: plasticity on/off switch of specific weight matrices IE
        :param e_u: plasticity on/off switch of specific weight matrices EE
        :param export: flag, if true in click, export
        :param export_dir_path: export file directory
        :param export_file_name: export file name, defaults to short description
        """
        self.short_description = short_description
        self.export = export
        if export:
            if export_file_name is None:
                export_file_name = '%s_data.p' % short_description
            if export_dir_path is None:
                raise Exception('Network: invalid export_dir_path: %s' % export_dir_path)
            self.export_file_path = export_dir_path + '/' + export_file_name
        else:
            self.export_file_path = None
        self.random_seed = random_seed
        self.online_saves = online_saves
        self.run_analysis = run_analysis
        self.T_train=int(T_train)
        self.T = self.T_train

        self.T_seq_train = T_seq_train
        self.T_seq = self.T_seq_train
        self.T_fine_train=int(T_fine_train)
        self.T_fine = self.T_fine_train

        self.N_pc = N_pc
        self.N_pv = N_pv
        self.N_L4 = N_L4

        self.gain = gain
        self.bias = bias
        self.n_exp = n_exp

        self.W_EE_norm = W_EE_norm
        self.W_EI_norm = W_EI_norm
        self.W_IE_norm = W_IE_norm
        self.W_II_norm = W_II_norm
        self.W_EF_norm = W_EF_norm
        self.W_IF_norm = W_IF_norm

        self.e_x_pc = 1 / e_x_pc_inv
        self.e_x_pv = 1 / e_x_pv_inv
        self.e_y = e_y

        self.std_W_EL = std_W_EL
        self.std_W_IL = std_W_IL
        self.std_W_II = std_W_II
        self.std_W_EI = std_W_EI
        self.std_W_IE = std_W_IE
        self.std_W_EE = std_W_EE

        self.tuned_init = tuned_init
        self.sigma_ori = sigma_ori

        self.input_sigma = input_sigma
        self.input_amp = input_amp

        self.l_norm = l_norm
        self.joint_norm = joint_norm
        self.lateral_norm = lateral_norm

        self.e = e

        self.e_w = e_w
        self.e_k = e_k
        self.e_p = e_p
        self.e_m = e_m
        self.e_q = e_q
        self.e_u = e_u

        self.initialize()

    def initialize(self):
        self.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d--%H-%M-%S')


        np.random.seed(self.random_seed)
        ###############################################################################
        self.accuracy = int(np.ceil(self.T / 10 ** 4))  # monitor recording variables every 'accuracy' iterations,
        # i.e., 10**x sample points in total
        self.accuracy_states = self.accuracy * 40  # monitor full network network every 'accuracy_states' iterations

        self.e_w_EE = self.e * 0.10 * 10 ** (-7)   # weight learning rate for excitatory connections onto excitatory neurons
        self.e_w_EI = self.e * 0.20 * 10 ** (-7)   # weight learning rate for inhibitory connections onto excitatory neurons

        self.e_w_IE = self.e * 0.15 * 10 ** (-7)   # weight learning rate for excitatory connections onto inhibitory neurons
        self.e_w_II = self.e * 0.25 * 10 ** (-7)        # weight learning rate for inhibitory connections onto inhibitory neurons

        # preallocate

        # L4 input
        self.y_L4 = np.zeros((self.N_L4,))  # L4 input
        self.y_L4_mean_init = 0.1
        self.y_mean_L4 = np.zeros((self.N_L4,)) + self.y_L4_mean_init

        # L2/3 PYR principal cells (pc)
        self.y_pc = np.zeros((self.N_pc,))
        self.y_0_pc = np.zeros((self.N_pc,)) + 1
        self.y_mean_pc = np.ones((self.N_pc,)) * 1
        self.y_var_pc = np.ones((self.N_pc,)) * 0.1
        self.x_pc = np.zeros((self.N_pc,))
        self.x_e_pc = np.zeros((self.N_pc,))
        self.x_i_pc = np.zeros((self.N_pc,))
        self.x_e_mean_pc = np.zeros((self.N_pc,))
        self.x_i_mean_pc = np.zeros((self.N_pc,))
        self.x_mean_pc = np.zeros((self.N_pc,))
        self.a_pc = np.zeros((self.N_pc,)) + self.gain
        self.b_pc = np.zeros((self.N_pc,)) + self.bias
        self.n_pc = np.zeros((self.N_pc,)) + self.n_exp
        self.W_EF_norms = np.zeros((self.N_pc,)) + self.W_EF_norm
        self.W_EI_norms = np.zeros((self.N_pc,)) + self.W_EI_norm
        self.W_EE_norms = np.zeros((self.N_pc,)) + self.W_EE_norm

        # L2/3 PV inhibitory cells (pv)
        self.y_pv = np.zeros((self.N_pv,))
        self.y_0_pv = np.zeros((self.N_pv,)) + 1
        self.y_mean_pv = np.ones((self.N_pv,)) * 1
        self.y_var_pv = np.ones((self.N_pv,)) * 0.1
        self.x_pv = np.zeros((self.N_pv,))
        self.x_e_pv = np.zeros((self.N_pv,))
        self.x_i_pv = np.zeros((self.N_pv,))
        self.x_e_mean_pv = np.zeros((self.N_pv,))
        self.x_i_mean_pv = np.zeros((self.N_pv,))
        self.x_mean_pv = np.zeros((self.N_pv,))
        self.a_pv = np.zeros((self.N_pv,)) + self.gain
        self.b_pv = np.zeros((self.N_pv,)) + self.bias
        self.n_pv = np.zeros((self.N_pv,)) + self.n_exp
        self.W_IF_norms = np.zeros((self.N_pv,)) + self.W_IF_norm
        self.W_II_norms = np.zeros((self.N_pv,)) + self.W_II_norm
        self.W_IE_norms = np.zeros((self.N_pv,)) + self.W_IE_norm

        # initialize weights
        self.W = abs(np.random.normal(2 * self.std_W_EL, self.std_W_EL, (self.N_pc, self.N_L4)))  # (L4->PC)
        self.K = abs(np.random.normal(2 * self.std_W_IL, self.std_W_IL, (self.N_pv, self.N_L4)))  # (L4->PV)
        self.P = abs(np.random.normal(2 * self.std_W_II, self.std_W_II, (self.N_pv, self.N_pv)))  # (PV-|PV)
        self.M = abs(np.random.normal(2 * self.std_W_EI, self.std_W_EI, (self.N_pc, self.N_pv)))  # (PV-|PC)
        self.Q = abs(np.random.normal(2 * self.std_W_IE, self.std_W_IE, (self.N_pv, self.N_pc)))  # (PC->PV)
        self.U = abs(np.random.normal(2 * self.std_W_EE, self.std_W_EE, (self.N_pc, self.N_pc)))  # (PC->PC)

        ###############################################################################
        # history preallocation

        self.network_states = [[]] * (2 + int(self.T / self.accuracy_states))  # preallocate memory to track network state
        self.count = int(self.T / self.accuracy)  # number of entries in history arrays
        self.t_hist = np.zeros((self.count,))

        # recorded from all neurons

        # PC
        self.x_pc_hist = np.zeros((self.count, self.N_pc))
        self.x_e_pc_hist = np.zeros((self.count, self.N_pc))
        self.x_i_pc_hist = np.zeros((self.count, self.N_pc))
        self.x_e_mean_pc_hist = np.zeros((self.count, self.N_pc))
        self.x_i_mean_pc_hist = np.zeros((self.count, self.N_pc))
        self.b_pc_hist = np.zeros((self.count, self.N_pc))
        self.a_pc_hist = np.zeros((self.count, self.N_pc))
        self.n_pc_hist = np.zeros((self.count, self.N_pc))
        self.W_EF_norms_hist = np.zeros((self.count, self.N_pc))
        self.W_EI_norms_hist = np.zeros((self.count, self.N_pc))
        self.W_EE_norms_hist = np.zeros((self.count, self.N_pc))
        self.y_pc_hist = np.zeros((self.count, self.N_pc))
        self.y_mean_pc_hist = np.zeros((self.count, self.N_pc))
        self.y_var_pc_hist = np.zeros((self.count, self.N_pc))
        self.y_0_pc_hist = np.zeros((self.count, self.N_pc))

        # PV
        self.x_pv_hist = np.zeros((self.count, self.N_pv))
        self.x_e_pv_hist = np.zeros((self.count, self.N_pv))
        self.x_i_pv_hist = np.zeros((self.count, self.N_pv))
        self.x_e_mean_pv_hist = np.zeros((self.count, self.N_pv))
        self.x_i_mean_pv_hist = np.zeros((self.count, self.N_pv))
        self.b_pv_hist = np.zeros((self.count, self.N_pv))
        self.a_pv_hist = np.zeros((self.count, self.N_pv))
        self.n_pv_hist = np.zeros((self.count, self.N_pv))
        self.W_IF_norms_hist = np.zeros((self.count, self.N_pv))
        self.W_II_norms_hist = np.zeros((self.count, self.N_pv))
        self.W_IE_norms_hist = np.zeros((self.count, self.N_pv))
        self.y_pv_hist = np.zeros((self.count, self.N_pv))
        self.y_mean_pv_hist = np.zeros((self.count, self.N_pv))
        self.y_var_pv_hist = np.zeros((self.count, self.N_pv))
        self.y_0_pv_hist = np.zeros((self.count, self.N_pv))

        self.y_mean_L4_hist = np.zeros((self.count, self.N_L4))

        # recorded from one neuron
        self.W_hist = np.zeros((self.count, self.N_L4))  # (L4->PC)
        self.K_hist = np.zeros((self.count, self.N_L4))  # (L4->PV)
        self.P_hist = np.zeros((self.count, self.N_pv))  # (PV-|PV)
        self.M_hist = np.zeros((self.count, self.N_pv))  # (PV-|PC)
        self.Q_hist = np.zeros((self.count, self.N_pc))  # (PC->PV)
        self.U_hist = np.zeros((self.count, self.N_pc))  # (PC->PC)

        # recorded from all neurons
        self.W_norm_hist = np.zeros((self.count, self.N_pc))  # (L4->PC)
        self.K_norm_hist = np.zeros((self.count, self.N_pv))  # (L4->PV)
        self.P_norm_hist = np.zeros((self.count, self.N_pv))  # (PV-|PV)
        self.M_norm_hist = np.zeros((self.count, self.N_pc))  # (PV-|PC)
        self.Q_norm_hist = np.zeros((self.count, self.N_pv))  # (PC->PV)
        self.U_norm_hist = np.zeros((self.count, self.N_pc))  # (PC->PC)

        # record variables every iteration during last 'T_fine' iterations

        # PC
        self.x_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.x_e_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.x_i_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.x_e_mean_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.x_i_mean_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.b_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.a_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.n_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.W_EF_norms_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.W_EI_norms_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.W_EE_norms_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.y_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.y_mean_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.y_var_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))
        self.y_0_pc_hist_fine = np.zeros((self.T_fine, self.N_pc))

        # PV
        self.x_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.x_e_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.x_i_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.x_e_mean_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.x_i_mean_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.b_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.a_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.n_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.W_IF_norms_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.W_II_norms_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.W_IE_norms_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.y_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.y_mean_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.y_var_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))
        self.y_0_pv_hist_fine = np.zeros((self.T_fine, self.N_pv))

        self.y_L4_hist_fine = np.zeros((self.T_fine, self.N_L4))
        self.y_mean_L4_hist_fine = np.zeros((self.T_fine, self.N_L4))

        # recorded from one neuron
        self.W_hist_fine = np.zeros((self.T_fine, self.N_pc, self.N_L4))  # (L4->PC)
        self.K_hist_fine = np.zeros((self.T_fine, self.N_pv, self.N_L4))  # (L4->PV)
        self.P_hist_fine = np.zeros((self.T_fine, self.N_pv, self.N_pv))  # (PV-|PV)
        self.M_hist_fine = np.zeros((self.T_fine, self.N_pc, self.N_pv))  # (PV-|PC)
        self.Q_hist_fine = np.zeros((self.T_fine, self.N_pv, self.N_pc))  # (PC->PV)
        self.U_hist_fine = np.zeros((self.T_fine, self.N_pc, self.N_pc))  # (PC->PC)

        # initialize tuned weights
        if self.tuned_init:
            self.d_theta_in = 180 / self.N_L4
            self.d_theta_pc = 180 / self.N_pc
            self.d_theta_pv = 180 / self.N_pv
            self.theta_pc = np.arange(0, 180, self.d_theta_pc)  # tuning peaks of pc neurons
            self.theta_pv = np.arange(0, 180, self.d_theta_pv)  # tuning peaks of pv neurons
            self.theta_in = np.arange(0, 180, self.d_theta_in)  # tuning peaks of in neurons

            self.W = fn.gaussian(self.theta_pc, self.theta_in, self.input_sigma)  # weight connectivity kernels
            self.K = fn.gaussian(self.theta_pv, self.theta_in, self.input_sigma)
            self.U = fn.gaussian(self.theta_pc, self.theta_pc, self.sigma_ori)
            self.Q = fn.gaussian(self.theta_pv, self.theta_pc, self.sigma_ori)
            self.M = fn.gaussian(self.theta_pc, self.theta_pv, self.sigma_ori)
            self.P = fn.gaussian(self.theta_pv, self.theta_pv, self.sigma_ori)

        self.norm_W = (np.sum(self.W ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # EF
        self.norm_K = (np.sum(self.K ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # IF
        self.norm_U = (np.sum(self.U ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # EE
        self.norm_M = (np.sum(self.M ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # EI
        self.norm_Q = (np.sum(self.Q ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # IE
        self.norm_P = (np.sum(self.P ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # II

        # after initialization, normalize input streams
        # normalize all excitatory inputs (lateral and ffwd.) together
        if self.joint_norm:
            self.W_EE_norms = self.W_EE_norms + self.W_EF_norms
            self.W_IE_norms = self.W_IE_norms + self.W_IF_norms

        # first establish feedforward and lateral weight norms as defined in config file (with small lateral E weights)
        self.W[self.norm_W != 0, :] *= (self.W_EF_norms[self.norm_W != 0, np.newaxis] / self.norm_W[self.norm_W != 0, np.newaxis])
        self.M[self.norm_M != 0, :] *= (self.W_EI_norms[self.norm_M != 0, np.newaxis] / self.norm_M[self.norm_M != 0, np.newaxis])
        self.U[self.norm_U != 0, :] *= (self.W_EE_norms[self.norm_U != 0, np.newaxis] / self.norm_U[self.norm_U != 0, np.newaxis])

        self.K[self.norm_K != 0, :] *= (self.W_IF_norms[self.norm_K != 0, np.newaxis] / self.norm_K[self.norm_K != 0, np.newaxis])
        self.P[self.norm_P != 0, :] *= (self.W_II_norms[self.norm_P != 0, np.newaxis] / self.norm_P[self.norm_P != 0, np.newaxis])
        self.Q[self.norm_Q != 0, :] *= (self.W_IE_norms[self.norm_Q != 0, np.newaxis] / self.norm_Q[self.norm_Q != 0, np.newaxis])

        self.norm_W = (np.sum(self.W ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # EF
        self.norm_K = (np.sum(self.K ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # IF
        self.norm_U = (np.sum(self.U ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # EE
        self.norm_M = (np.sum(self.M ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # EI
        self.norm_Q = (np.sum(self.Q ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # IE
        self.norm_P = (np.sum(self.P ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # II

        self.norm_WU = (self.norm_W ** self.l_norm + self.norm_U ** self.l_norm) ** (1 / self.l_norm)  # EE
        self.norm_KQ = (self.norm_K ** self.l_norm + self.norm_Q ** self.l_norm) ** (1 / self.l_norm)  # IE

        # normalization of exciatory PC input
        if self.joint_norm:
            self.W *= (self.W_EE_norms[:, np.newaxis] / self.norm_WU[:, np.newaxis])
            self.U *= (self.W_EE_norms[:, np.newaxis] / self.norm_WU[:, np.newaxis])
        else:
            self.W *= (self.W_EF_norms[:, np.newaxis] / self.norm_W[:, np.newaxis])
            self.U *= (self.W_EE_norms[:, np.newaxis] / self.norm_U[:, np.newaxis])

        # normalization of inhibitory PC input
        self.M[self.norm_M != 0, :] *= (
                self.W_EI_norms[self.norm_M != 0, np.newaxis] / self.norm_M[self.norm_M != 0, np.newaxis])  # do not scale 0-norm input

        # joint normalization of all exciatory PV input
        if self.joint_norm:
            self.K *= (self.W_IE_norms[:, np.newaxis] / self.norm_KQ[:, np.newaxis])
            self.Q *= (self.W_IE_norms[:, np.newaxis] / self.norm_KQ[:, np.newaxis])
        else:
            self.K *= (self.W_IF_norms[:, np.newaxis] / self.norm_K[:, np.newaxis])
            self.Q *= (self.W_IE_norms[:, np.newaxis] / self.norm_Q[:, np.newaxis])

        # normalization of inhibitory PV input
        self.P[self.norm_P != 0, :] *= (
                self.W_II_norms[self.norm_P != 0, np.newaxis] / self.norm_P[self.norm_P != 0, np.newaxis])  # do not scale 0-norm input

        # gather network state variables in a list for later saving
        self.network = [self.W, self.K, self.P, self.M, self.Q, self.U,
                   self.a_pc, self.b_pc, self.n_pc, self.W_EF_norms, self.W_EI_norms, self.W_EE_norms, self.y_mean_pc,
                        self.y_var_pc, self.y_0_pc,
                   self.a_pv, self.b_pv, self.n_pv, self.W_IF_norms, self.W_II_norms, self.W_IE_norms, self.y_mean_pv,
                        self.y_var_pv, self.y_0_pv]

        self.hist = [self.x_pc_hist, self.x_e_pc_hist, self.x_i_pc_hist, self.x_e_mean_pc_hist, self.x_i_mean_pc_hist, self.b_pc_hist, self.a_pc_hist,
                self.n_pc_hist,
                self.W_EF_norms_hist, self.W_EI_norms_hist, self.W_EE_norms_hist, self.y_pc_hist, self.y_mean_pc_hist, self.y_var_pc_hist,
                self.y_0_pc_hist,
                self.x_pv_hist, self.x_e_pv_hist, self.x_i_pv_hist, self.x_e_mean_pv_hist, self.x_i_mean_pv_hist, self.b_pv_hist, self.a_pv_hist,
                self.n_pv_hist,
                self.W_IF_norms_hist, self.W_II_norms_hist, self.W_IE_norms_hist, self.y_pv_hist, self.y_mean_pv_hist, self.y_var_pv_hist,
                self.y_0_pv_hist,
                self.W_hist, self.K_hist, self.P_hist, self.M_hist, self.Q_hist, self.U_hist,
                self.W_norm_hist, self.K_norm_hist, self.P_norm_hist, self.M_norm_hist, self.Q_norm_hist, self.U_norm_hist]

        self.hist_fine = [self.x_pc_hist_fine, self.x_e_pc_hist_fine, self.x_i_pc_hist_fine, self.x_e_mean_pc_hist_fine, self.x_i_mean_pc_hist_fine,
                     self.b_pc_hist_fine, self.a_pc_hist_fine, self.n_pc_hist_fine, self.W_EF_norms_hist_fine, self.W_EI_norms_hist_fine,
                     self.W_EE_norms_hist_fine, self.y_pc_hist_fine, self.y_mean_pc_hist_fine, self.y_var_pc_hist_fine, self.y_0_pc_hist_fine,
                     self.x_pv_hist_fine, self.x_e_pv_hist_fine, self.x_i_pv_hist_fine, self.x_e_mean_pv_hist_fine, self.x_i_mean_pv_hist_fine,
                     self.b_pv_hist_fine, self.a_pv_hist_fine, self.n_pv_hist_fine, self.W_IF_norms_hist_fine, self.W_II_norms_hist_fine,
                     self.W_IE_norms_hist_fine, self.y_pv_hist_fine, self.y_mean_pv_hist_fine, self.y_var_pv_hist_fine, self.y_0_pv_hist_fine,
                     self.y_L4_hist_fine,
                     self.W_hist_fine, self.K_hist_fine, self.P_hist_fine, self.M_hist_fine, self.Q_hist_fine, self.U_hist_fine]
        self.param = [self.e_x_pc, self.e_x_pv, self.e_y, self.e_w_EE, self.e_w_IE, self.e_w_EI, self.e_w_II,
                 self.N_L4, self.N_pc, self.N_pv, self.input_amp, self.input_sigma]

    def train(self):
        self.T = self.T_train
        self.T_fine = self.T_fine_train
        self.T_seq = self.T_seq_train
        self.start_time = time.time()
        self.print_time = 0  # needed for check how much time has passed since the last time progress has been printed

        for t in tqdm(range(self.T)):

            # load input patch
            if t % self.T_seq == 0:  # switch input from L4 every 'T_seq' iterations
                if t >= self.T - 1000:  # in last 1k iterations sweep over whole input space to measure orientation tunings
                    self.orientation = ((t / self.T_seq) % 25) / 25 * np.pi  # sweep over all orientation within 25 trials
                    self.y_L4 = inputs.get_input(self.N_L4, theta=self.orientation, sigma=self.input_sigma * np.pi / 180, amp=self.input_amp)
                else:
                    self.y_L4 = inputs.get_input(self.N_L4, sigma=self.input_sigma * np.pi / 180, amp=self.input_amp)
            # Average mean L4 input for timestep t:
            # timescale of online exponential weighted average * (-current L4 input + new L4 input)
            self.y_mean_L4 += self.e_y * (-self.y_mean_L4 + self.y_L4)

            self.x_e_pc = np.dot(self.W, self.y_L4) + np.dot(self.U, self.y_pc) # EF + EE input
            self.x_i_pc = np.dot(self.M, self.y_pv) # EI input

            self.x_e_pv = np.dot(self.K, self.y_L4) + np.dot(self.Q, self.y_pc) # IF + IE input
            self.x_i_pv = np.dot(self.P, self.y_pv) # II input

            self.x_pc += self.e_x_pc * (-self.x_pc + self.x_e_pc - self.x_i_pc) # membrane timescale * total input onto excitatory cells (PC)
            self.x_pv += self.e_x_pv * (-self.x_pv + self.x_e_pv - self.x_i_pv) # membrane timescale * total input on inhibitory cells (PV)

            self.y_pc = self.a_pc * ((abs(self.x_pc) + self.x_pc) / 2) ** self.n_pc # total activation of all PC cells
            self.y_pv = self.a_pv * ((abs(self.x_pv) + self.x_pv) / 2) ** self.n_pv # total activation of all PV cells

            ###############################################################################

            # running averages and variances
            self.y_mean_pc_old = self.y_mean_pc
            self.y_mean_pc += self.e_y * (-self.y_mean_pc + self.y_pc)
            self.y_var_pc += self.e_y * (-self.y_var_pc + (self.y_pc - self.y_mean_pc_old) * (self.y_pc - self.y_mean_pc))
            self.x_e_mean_pc += self.e_y * (-self.x_e_mean_pc + self.x_e_pc)
            self.x_i_mean_pc += self.e_y * (-self.x_i_mean_pc + self.x_i_pc)

            self.y_mean_pv_old = self.y_mean_pv
            self.y_mean_pv += self.e_y * (-self.y_mean_pv + self.y_pv)
            self.y_var_pv += self.e_y * (-self.y_var_pv + (self.y_pv - self.y_mean_pv_old) * (self.y_pv - self.y_mean_pv))
            self.x_e_mean_pv += self.e_y * (-self.x_e_mean_pv + self.x_e_pv)
            self.x_i_mean_pv += self.e_y * (-self.x_i_mean_pv + self.x_i_pv)

            # weight adaption
            self.W += self.e_w * self.e_w_EE * np.outer(self.y_pc, self.y_L4)  # L4->PC
            self.K += self.e_k * self.e_w_IE * np.outer(self.y_pv, self.y_L4)  # L4->PV
            self.P += self.e_p * self.e_w_II * np.outer(self.y_pv, self.y_pv)  # PV-|PV
            self.M += self.e_m * self.e_w_EI * np.outer(self.y_pc, self.y_pv)  # PV-|PC
            self.Q += self.e_q * self.e_w_IE * np.outer(self.y_pv, self.y_pc)  # PC->PV
            self.U += self.e_u * self.e_w_EE * np.outer(self.y_pc, self.y_pc)  # PC->PC

            # enforce Dale's law: a neuron transmits the same number of neurotransmitters at all of its postsyn connections
            self.W[self.W < 0] = 0  # L4->PC
            self.K[self.K < 0] = 0  # L4->PV
            self.P[self.P < 0] = 0  # PV-|PV
            self.M[self.M < 0] = 0  # PV-|PC
            self.Q[self.Q < 0] = 0  # PC->PV
            self.U[self.U < 0] = 0  # PC->PV

            # weight normalization
            self.norm_W = (np.sum(self.W ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # EF
            self.norm_K = (np.sum(self.K ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # IF
            self.norm_U = (np.sum(self.U ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # EE
            self.norm_M = (np.sum(self.M ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # EI
            self.norm_Q = (np.sum(self.Q ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # IE
            self.norm_P = (np.sum(self.P ** self.l_norm, axis=1)) ** (1 / self.l_norm)  # II

            self.norm_WU = (self.norm_W ** self.l_norm + self.norm_U ** self.l_norm) ** (1 / self.l_norm)  # EE
            self.norm_KQ = (self.norm_K ** self.l_norm + self.norm_Q ** self.l_norm) ** (1 / self.l_norm)  # IE

            # normalization of exciatory PC input
            if self.joint_norm:
                self.W *= (self.W_EE_norms[:, np.newaxis] / self.norm_WU[:, np.newaxis]) ** self.e_w
                self.U *= (self.W_EE_norms[:, np.newaxis] / self.norm_WU[:, np.newaxis]) ** self.e_u
            else:
                self.W *= (self.W_EF_norms[:, np.newaxis] / self.norm_W[:, np.newaxis]) ** self.e_w
                self.U *= (self.W_EE_norms[:, np.newaxis] / self.norm_U[:, np.newaxis]) ** self.e_u

            # normalization of inhibitory PC input
            self.M[self.norm_M != 0, :] *= (self.W_EI_norms[self.norm_M != 0, np.newaxis] / self.norm_M[
                self.norm_M != 0, np.newaxis]) ** self.e_m  # do not scale 0-norm weight vectors

            # normalization of exciatory PV input
            if self.joint_norm:
                self.K *= (self.W_IE_norms[:, np.newaxis] / self.norm_KQ[:, np.newaxis]) ** self.e_k
                self.Q *= (self.W_IE_norms[:, np.newaxis] / self.norm_KQ[:, np.newaxis]) ** self.e_q
            else:
                self.K *= (self.W_IF_norms[:, np.newaxis] / self.norm_K[:, np.newaxis]) ** self.e_k
                self.Q *= (self.W_IE_norms[:, np.newaxis] / self.norm_Q[:, np.newaxis]) ** self.e_q

            # normalization of inhibitory PV input
            self.P[self.norm_P != 0, :] *= (self.W_II_norms[self.norm_P != 0, np.newaxis] / self.norm_P[
                self.norm_P != 0, np.newaxis]) ** self.e_p  # do not scale 0-norm weight vectors

            ##############################################################################################################################################################
            ##############################################################################################################################################################
            # BOOKKEEPING

            # record variables of interest
            if t % self.accuracy == 0:
                ind = int(t / self.accuracy)
                self.t_hist[ind] = t

                self.y_mean_L4_hist[ind] = self.y_mean_L4

                self.x_pc_hist[ind] = self.x_pc
                self.x_e_pc_hist[ind] = self.x_e_pc
                self.x_i_pc_hist[ind] = self.x_i_pc
                self.x_e_mean_pc_hist[ind] = self.x_e_mean_pc
                self.x_i_mean_pc_hist[ind] = self.x_i_mean_pc
                self.b_pc_hist[ind] = self.b_pc
                self.a_pc_hist[ind] = self.a_pc
                self.n_pc_hist[ind] = self.n_pc
                self.W_EE_norms_hist[ind] = self.W_EE_norms
                self.y_pc_hist[ind] = self.y_pc
                self.y_mean_pc_hist[ind] = self.y_mean_pc
                self.y_var_pc_hist[ind] = self.y_var_pc
                self.y_0_pc_hist[ind] = self.y_0_pc

                self.x_pv_hist[ind] = self.x_pv
                self.x_e_pv_hist[ind] = self.x_e_pv
                self.x_i_pv_hist[ind] = self.x_i_pv
                self.x_e_mean_pv_hist[ind] = self.x_e_mean_pv
                self.x_i_mean_pv_hist[ind] = self.x_i_mean_pv
                self.b_pv_hist[ind] = self.b_pv
                self.a_pv_hist[ind] = self.a_pv
                self.n_pv_hist[ind] = self.n_pv
                self.W_IE_norms_hist[ind] = self.W_IE_norms
                self.y_pv_hist[ind] = self.y_pv
                self.y_mean_pv_hist[ind] = self.y_mean_pv
                self.y_var_pv_hist[ind] = self.y_var_pv
                self.y_0_pv_hist[ind] = self.y_0_pv

                # record weight data from one neuron
                self.W_hist[ind] = self.W[0]
                self.W_norm_hist[ind] = np.sum(self.W, axis=1)[0]
                self.K_hist[ind] = self.K[0]
                self.K_norm_hist[ind] = np.sum(self.K, axis=1)[0]
                self.P_hist[ind] = self.P[0]
                self.P_norm_hist[ind] = np.sum(self.P, axis=1)[0]
                self.M_hist[ind] = self.M[0]
                self.M_norm_hist[ind] = np.sum(self.M, axis=1)[0]
                self.Q_hist[ind] = self.Q[0]
                self.Q_norm_hist[ind] = np.sum(self.Q, axis=1)[0]
                self.U_hist[ind] = self.U[0]
                self.U_norm_hist[ind] = np.sum(self.U, axis=1)[0]

            # during last 'T_fine' iterations, record variables at every iteration
            if t >= self.T - self.T_fine:
                ind = t - (self.T - self.T_fine)

                self.x_pc_hist_fine[ind] = self.x_pc
                self.x_e_pc_hist_fine[ind] = self.x_e_pc
                self.x_i_pc_hist_fine[ind] = self.x_i_pc
                self.x_e_mean_pc_hist_fine[ind] = self.x_e_mean_pc
                self.x_i_mean_pc_hist_fine[ind] = self.x_i_mean_pc
                self.b_pc_hist_fine[ind] = self.b_pc
                self.a_pc_hist_fine[ind] = self.a_pc
                self.n_pc_hist_fine[ind] = self.n_pc
                self.W_EE_norms_hist_fine[ind] = self.W_EE_norms
                self.y_pc_hist_fine[ind] = self.y_pc
                self.y_mean_pc_hist_fine[ind] = self.y_mean_pc  # average pyramidal cell hist
                self.y_var_pc_hist_fine[ind] = self.y_var_pc
                self.y_0_pc_hist_fine[ind] = self.y_0_pc

                self.x_pv_hist_fine[ind] = self.x_pv
                self.x_e_pv_hist_fine[ind] = self.x_e_pv
                self.x_i_pv_hist_fine[ind] = self.x_i_pv
                self.x_e_mean_pv_hist_fine[ind] = self.x_e_mean_pv
                self.x_i_mean_pv_hist_fine[ind] = self.x_i_mean_pv
                self.b_pv_hist_fine[ind] = self.b_pv
                self.a_pv_hist_fine[ind] = self.a_pv
                self.n_pv_hist_fine[ind] = self.n_pv
                self.W_IE_norms_hist_fine[ind] = self.W_IE_norms
                self.y_pv_hist_fine[ind] = self.y_pv
                self.y_mean_pv_hist_fine[ind] = self.y_mean_pv  # interneuron history
                self.y_var_pv_hist_fine[ind] = self.y_var_pv
                self.y_0_pv_hist_fine[ind] = self.y_0_pv

                self.y_L4_hist_fine[ind] = self.y_L4
                self.y_mean_L4_hist_fine[ind] = self.y_mean_L4  # mean activity of layer 4 inputs

                self.W_hist_fine[ind, :, :] = self.W
                self.K_hist_fine[ind, :, :] = self.K
                self.P_hist_fine[ind, :, :] = self.P
                self.M_hist_fine[ind, :, :] = self.M
                self.Q_hist_fine[ind, :, :] = self.Q
                self.U_hist_fine[ind, :, :] = self.U

            if t % self.accuracy_states == 0:
                ind = int(t / self.accuracy_states) + 1
                self.network_states[ind] = copy.deepcopy([t] + self.network)

            # save data every 10% of runtime
            if ((t + 1) % int(self.T / 10) == 0) and self.online_saves and self.export:
                self.export_data()

        print("-----------------")

        self.network_states[-1] = copy.deepcopy([t] + self.network)
        if self.export:
            self.export_data()

        self.t = t
        self.runtime = time.time() - self.start_time
        print("runtime: ", end='', flush=True)
        print(int(np.floor(self.runtime / 60)), end='')
        print("min ", end='')
        print(int(self.runtime - np.floor(self.runtime / 60) * 60), end='')
        print("sec")

        print("-----------------")

    def export_data(self, export_file_path=None):
        if export_file_path is None:
            if self.export_file_path is None:
                raise Exception('Network.export_data: no export_file_path specified')
            export_file_path = self.export_file_path
        io.save([self.network_states, self.hist, self.hist_fine, self.param], export_file_path)
        print("network saved to path: %s" % self.export_file_path)

    def load_data(self, data_file_path):
        self.loaded_data = io.load(data_file_path)
        [self.t, self.W, self.K, self.P, self.M, self.Q, self.U,
         self.a_pc, self.b_pc, self.n_pc, self.W_EF_norms, self.W_EI_norms, self.W_EE_norms, self.y_mean_pc, self.y_var_pc, self.y_0_pc,
         self.a_pv, self.b_pv, self.n_pv, self.W_IF_norms, self.W_II_norms, self.W_IE_norms, self.y_mean_pv, self.y_var_pv, self.y_0_pv] = self.loaded_data[0][-1]

        """
        ?
        n_pc = 80
        n_pv = 20
        W_EF_norms = W_EE_norms
        W_IF_norms = W_IE_norms
        """

        # load original hist
        [self.x_pc_hist, self.x_e_pc_hist, self.x_i_pc_hist, self.x_e_mean_pc_hist, self.x_i_mean_pc_hist, self.b_pc_hist, self.a_pc_hist, self.n_pc_hist,
         self.W_EF_norms_hist, self.W_EI_norms_hist, self.W_EE_norms_hist, self.y_pc_hist, self.y_mean_pc_hist, self.y_var_pc_hist, self.y_0_pc_hist,
         self.x_pv_hist, self.x_e_pv_hist, self.x_i_pv_hist, self.x_e_mean_pv_hist, self.x_i_mean_pv_hist, self.b_pv_hist, self.a_pv_hist, self.n_pv_hist,
         self.W_IF_norms_hist, self.W_II_norms_hist, self.W_IE_norms_hist, self.y_pv_hist, self.y_mean_pv_hist, self.y_var_pv_hist, self.y_0_pv_hist,
         self.W_hist, self.K_hist, self.P_hist, self.M_hist, self.Q_hist, self.U_hist,
         self.W_norm_hist, self.K_norm_hist, self.P_norm_hist, self.M_norm_hist, self.Q_norm_hist, self.U_norm_hist] = self.loaded_data[1]

        # load fine hist
        [self.x_pc_hist_fine, self.x_e_pc_hist_fine, self.x_i_pc_hist_fine, self.x_e_mean_pc_hist_fine,self.x_i_mean_pc_hist_fine,
         self.b_pc_hist_fine, self.a_pc_hist_fine, self.n_pc_hist_fine, self.W_EF_norms_hist_fine,self.W_EI_norms_hist_fine,
         self.W_EE_norms_hist_fine, self.y_pc_hist_fine, self.y_mean_pc_hist_fine, self.y_var_pc_hist_fine,self.y_0_pc_hist_fine,
         self.x_pv_hist_fine, self.x_e_pv_hist_fine, self.x_i_pv_hist_fine, self.x_e_mean_pv_hist_fine,
         self.x_i_mean_pv_hist_fine,self.b_pv_hist_fine, self.a_pv_hist_fine, self.n_pv_hist_fine, self.W_IF_norms_hist_fine,
         self.W_II_norms_hist_fine,self.W_IE_norms_hist_fine, self.y_pv_hist_fine, self.y_mean_pv_hist_fine, self.y_var_pv_hist_fine,
         self.y_0_pv_hist_fine,self.y_L4_hist_fine,self.W_hist_fine, self.K_hist_fine, self.P_hist_fine, self.M_hist_fine, self.Q_hist_fine, self.U_hist_fine] =self.loaded_data[2]

        # load original network parameters
        [self.e_x_pc, self.e_x_pv, self.e_y, self.e_w_EE, self.e_w_IE, self.e_w_EI, self.e_w_II,
         self.N_L4, self.N_pc, self.N_pv, self.input_amp, self.input_sigma] = self.loaded_data[3]

    def test(self):

        self.T_seq = 100  # how many iterations per stimulus
        self.n_probe = 100  # how many different stimuli to probe
        self.T = self.n_probe * self.T_seq
        self.T_fine = self.T

        # Define some variables

        # L4 input
        self.y_L4 = np.zeros((self.N_L4,))  # L4 input
        self.y_mean_L4 = np.zeros((self.N_L4,))

        # L2/3 PYR principal cells (pc)
        self.y_pc = np.zeros((self.N_pc,))
        self._pc = np.zeros((self.N_pc,))
        self.x_e_pc = np.zeros((self.N_pc,))
        self.x_i_pc = np.zeros((self.N_pc,))
        self.x_e_mean_pc = np.zeros((self.N_pc,))
        self.x_i_mean_pc = np.zeros((self.N_pc,))

        # L2/3 PV inhibitory cells (pv)
        self.y_pv = np.zeros((self.N_pv,))
        self.x_pv = np.zeros((self.N_pv,))
        self.x_e_pv = np.zeros((self.N_pv,))
        self.x_i_pv = np.zeros((self.N_pv,))
        self.x_e_mean_pv = np.zeros((self.N_pv,))
        self.x_i_mean_pv = np.zeros((self.N_pv,))

        ###############################################################################
        # history preallocation

        self.t_hist_fine = np.zeros((self.T,))
        self.theta_hist_fine = np.zeros((self.T,))

        # PC
        self.x_pc_hist_fine = np.zeros((self.T, self.N_pc))
        self.x_e_pc_hist_fine = np.zeros((self.T, self.N_pc))
        self.x_i_pc_hist_fine = np.zeros((self.T, self.N_pc))
        self.x_e_mean_pc_hist_fine = np.zeros((self.T, self.N_pc))
        self.x_i_mean_pc_hist_fine = np.zeros((self.T, self.N_pc))
        self.y_pc_hist_fine = np.zeros((self.T, self.N_pc))
        self.y_mean_pc_hist_fine = np.zeros((self.T, self.N_pc))
        self.y_var_pc_hist_fine = np.zeros((self.T, self.N_pc))

        # PV
        self.x_pv_hist_fine = np.zeros((self.T, self.N_pv))
        self.x_e_pv_hist_fine = np.zeros((self.T, self.N_pv))
        self.x_i_pv_hist_fine = np.zeros((self.T, self.N_pv))
        self.x_e_mean_pv_hist_fine = np.zeros((self.T, self.N_pv))
        self.x_i_mean_pv_hist_fine = np.zeros((self.T, self.N_pv))
        self.y_pv_hist_fine = np.zeros((self.T, self.N_pv))
        self.y_mean_pv_hist_fine = np.zeros((self.T, self.N_pv))
        self.y_var_pv_hist_fine = np.zeros((self.T, self.N_pv))

        # L4
        self.y_L4_hist_fine = np.zeros((self.T, self.N_L4))
        self.y_mean_L4_hist_fine = np.zeros((self.T, self.N_L4))

        #######################################################################################################################
        #######################################################################################################################
        #######################################################################################################################
        # ------------ SIM ------------------

        # for t in tqdm(range(T)):
        for t in range(self.T):

            # load input patch
            if t % self.T_seq == 0:  # switch input every 'T_seq' iterations
                self.orientation = (( t / self.T_seq) % self.n_probe) / self.n_probe * np.pi  # sweep over all orientations. Probe 'n_probe' orientations, 'T_seq' iterations per orientation
                self.y_L4 = inputs.get_input(self.N_L4, theta=self.orientation, sigma=self.input_sigma * np.pi / 180, amp=self.input_amp)

                # reset network activity
                self.y_pc *= 0
                self.y_pv *= 0
                self.x_e_pc *= 0
                self.x_i_pc *= 0
                self.x_e_pv *= 0
                self.x_i_pv *= 0

            self.x_e_pc += self.e_x_pc * (-self.x_e_pc + np.dot(self.W, self.y_L4) + np.dot(self.U, self.y_pc))
            self.x_i_pc += self.e_x_pc * (-self.x_i_pc + np.dot(self.M, self.y_pv))
            self.x_pc = self.x_e_pc - self.x_i_pc

            self.x_e_pv += self.e_x_pv * (-self.x_e_pv + np.dot(self.K, self.y_L4) + np.dot(self.Q, self.y_pc))
            self.x_i_pv += self.e_x_pv * (-self.x_i_pv + np.dot(self.P, self.y_pv))
            self.x_pv = self.x_e_pv - self.x_i_pv

            self.x_pc = self.x_pc.clip(min=0)  # clip negative membrane potential values to zero
            self.x_pv = self.x_pv.clip(min=0)

            self.y_pc = self.a_pc * (self.x_pc ** 2)  # get firing rates
            self.y_pv = self.a_pv * (self.x_pv ** 2)  # get firing rates

            ###############################################################################
            # running averages + variances
            self.y_mean_pc_old = self.y_mean_pc
            self.y_mean_pc += self.e_y * (-self.y_mean_pc + self.y_pc)
            self.y_var_pc += self.e_y * (-self.y_var_pc + (self.y_pc - self.y_mean_pc_old) * (self.y_pc - self.y_mean_pc))
            self.x_e_mean_pc += self.e_y * (-self.x_e_mean_pc + self.x_e_pc)
            self.x_i_mean_pc += self.e_y * (-self.x_i_mean_pc + self.x_i_pc)

            self.y_mean_pv_old = self.y_mean_pv
            self.y_mean_pv += self.e_y * (-self.y_mean_pv + self.y_pv)
            self.y_var_pv += self.e_y * (-self.y_var_pv + (self.y_pv - self.y_mean_pv_old) * (self.y_pv - self.y_mean_pv))
            self.x_e_mean_pv += self.e_y * (-self.x_e_mean_pv + self.x_e_pv)
            self.x_i_mean_pv += self.e_y * (-self.x_i_mean_pv + self.x_i_pv)

            ###############################################################################
            # record variables in every iterations

            self.t_hist_fine[t] = t
            self.theta_hist_fine[t] = self.orientation

            self.x_pc_hist_fine[t] = self.x_pc
            self.x_e_pc_hist_fine[t] = self.x_e_pc
            self.x_i_pc_hist_fine[t] = self.x_i_pc
            self.x_e_mean_pc_hist_fine[t] = self.x_e_mean_pc
            self.x_i_mean_pc_hist_fine[t] = self.x_i_mean_pc
            self.y_pc_hist_fine[t] = self.y_pc
            self.y_mean_pc_hist_fine[t] = self.y_mean_pc
            self.y_var_pc_hist_fine[t] = self.y_var_pc

            self.x_pv_hist_fine[t] = self.x_pv
            self.x_e_pv_hist_fine[t] = self.x_e_pv
            self.x_i_pv_hist_fine[t] = self.x_i_pv
            self.x_e_mean_pv_hist_fine[t] = self.x_e_mean_pv
            self.x_i_mean_pv_hist_fine[t] = self.x_i_mean_pv
            self.y_pv_hist_fine[t] = self.y_pv
            self.y_mean_pv_hist_fine[t] = self.y_mean_pv
            self.y_var_pv_hist_fine[t] = self.y_var_pv

            self.y_L4_hist_fine[t] = self.y_L4
            self.y_mean_L4_hist_fine[t] = self.y_mean_L4

        print("sim ready")

    def plot_test(self):
        # calculate tuning curves + E/I balance of all PV and PC cells
        # -------------------------------------

        # take average activity for each orientation (averaged over second half of trial T_seq to mitigate transient dynamics)
        self.x_e_pc_stim_avg = np.mean(
            np.reshape(self.x_e_pc_hist_fine, (int(self.T_fine / self.T_seq), self.T_seq, self.N_pc))[:, int(self.T_seq * 2 / 4):, :], axis=1)
        self.x_i_pc_stim_avg = np.mean(
            np.reshape(self.x_i_pc_hist_fine, (int(self.T_fine / self.T_seq), self.T_seq, self.N_pc))[:, int(self.T_seq * 2 / 4):, :], axis=1)
        self.y_pc_stim_avg = np.mean(np.reshape(self.y_pc_hist_fine, (int(self.T_fine / self.T_seq), self.T_seq, self.N_pc))[:, int(self.T_seq * 2 / 4):, :],
                                axis=1)

        self.x_e_pv_stim_avg = np.mean(
            np.reshape(self.x_e_pv_hist_fine, (int(self.T_fine / self.T_seq), self.T_seq, self.N_pv))[:, int(self.T_seq * 2 / 4):, :], axis=1)
        self.x_i_pv_stim_avg = np.mean(
            np.reshape(self.x_i_pv_hist_fine, (int(self.T_fine / self.T_seq), self.T_seq, self.N_pv))[:, int(self.T_seq * 2 / 4):, :], axis=1)
        self.y_pv_stim_avg = np.mean(np.reshape(self.y_pv_hist_fine, (int(self.T_fine / self.T_seq), self.T_seq, self.N_pv))[:, int(self.T_seq * 2 / 4):, :],
                                axis=1)

        self.y_L4_stim_avg = np.mean(np.reshape(self.y_L4_hist_fine, (int(self.T_fine / self.T_seq), self.T_seq, self.N_L4))[:, int(self.T_seq * 2 / 4):, :],
                                axis=1)

        self.y_limit = 1.1 * np.max([np.max(elem) for elem in [self.x_e_pc_stim_avg, self.x_i_pc_stim_avg, self.x_e_pv_stim_avg,
                                                          self.x_i_pv_stim_avg]])  # y_axis limit for plotting

        self.theta_stim_avg = self.theta_hist_fine[0::self.T_seq]

        # rearrange neurons according to preferred tunings
        self.sorted_pc_ids = np.argsort(np.argmax(self.y_pc_stim_avg, axis=0))
        self.sorted_pv_ids = np.argsort(np.argmax(self.y_pv_stim_avg, axis=0))
        self.sorted_L4_ids = np.argsort(np.argmax(self.y_L4_stim_avg, axis=0))

        # save tuning curve peak of pc and pc neuron_class
        # -------------------------------------
        self.pc_tuning_peak = self.theta_stim_avg[np.argmax(self.y_pc_stim_avg, axis=0)]
        self.pv_tuning_peak = self.theta_stim_avg[np.argmax(self.y_pv_stim_avg, axis=0)]
        self.L4_tuning_peak = self.theta_stim_avg[np.argmax(self.y_L4_stim_avg, axis=0)]

        # plot average tuning curves
        # -------------------------------------

        # PC
        # add periods before cutting around tuning peak
        self.theta_stim_avg_centered = np.concatenate((self.theta_stim_avg - np.pi, self.theta_stim_avg, self.theta_stim_avg + np.pi),
                                                 axis=0)
        self.y_pc_stim_avg_centered = np.concatenate((self.y_pc_stim_avg, self.y_pc_stim_avg, self.y_pc_stim_avg), axis=0)
        self.x_e_pc_stim_avg_centered = np.concatenate((self.x_e_pc_stim_avg, self.x_e_pc_stim_avg, self.x_e_pc_stim_avg), axis=0)
        self.x_i_pc_stim_avg_centered = np.concatenate((self.x_i_pc_stim_avg, self.x_i_pc_stim_avg, self.x_i_pc_stim_avg), axis=0)

        # gather indices for cutting around tuning peak for each neuron
        self.start = (self.n_probe + np.argmax(self.y_pc_stim_avg, axis=0) - self.n_probe / 2).astype(int)
        self.stop = (self.n_probe / 2 + np.argmax(self.y_pc_stim_avg, axis=0) + self.n_probe).astype(int)
        idx = np.array([[i for i in np.arange(self.start[j], self.stop[j])] for j in np.arange(self.start.shape[0])])

        # cut region around tuning peak
        self.theta_delta_pc = self.theta_stim_avg_centered[idx] - self.pc_tuning_peak[:, np.newaxis]
        self.y_pc_stim_avg_centered = self.y_pc_stim_avg_centered[idx.T, np.arange(self.N_pc)]
        self.x_e_pc_stim_avg_centered = self.x_e_pc_stim_avg_centered[idx.T, np.arange(self.N_pc)]
        self.x_i_pc_stim_avg_centered = self.x_i_pc_stim_avg_centered[idx.T, np.arange(self.N_pc)]

        # PV
        # add periods before cutting around tuning peak
        self.y_pv_stim_avg_centered = np.concatenate((self.y_pv_stim_avg, self.y_pv_stim_avg, self.y_pv_stim_avg),
                                                axis=0)  # add periods before cutting around tuning peak
        self.x_e_pv_stim_avg_centered = np.concatenate((self.x_e_pv_stim_avg, self.x_e_pv_stim_avg, self.x_e_pv_stim_avg), axis=0)
        self.x_i_pv_stim_avg_centered = np.concatenate((self.x_i_pv_stim_avg, self.x_i_pv_stim_avg, self.x_i_pv_stim_avg), axis=0)

        # gather indices for cutting around tuning peak for each neuron
        self.start = (self.n_probe + np.argmax(self.y_pv_stim_avg, axis=0) - self.n_probe / 2).astype(int)
        self.stop = (self.n_probe / 2 + np.argmax(self.y_pv_stim_avg, axis=0) + self.n_probe).astype(int)
        idx = np.array([[i for i in np.arange(self.start[j], self.stop[j])] for j in np.arange(self.start.shape[0])])

        # cut region around tuning peak
        self.theta_delta_pv = self.theta_stim_avg_centered[idx] - self.pv_tuning_peak[:, np.newaxis]
        self.y_pv_stim_avg_centered = self.y_pv_stim_avg_centered[idx.T, np.arange(self.N_pv)]
        self.x_e_pv_stim_avg_centered = self.x_e_pv_stim_avg_centered[idx.T, np.arange(self.N_pv)]
        self.x_i_pv_stim_avg_centered = self.x_i_pv_stim_avg_centered[idx.T, np.arange(self.N_pv)]

        # plot average E and I input to an E neuron
        # -------------------------------------
        self.x_data = [np.mean(self.theta_delta_pc, axis=0) * 180 / np.pi, np.mean(self.theta_delta_pc, axis=0) * 180 / np.pi]
        self.y_data = [np.mean(self.x_e_pc_stim_avg_centered, axis=1), np.mean(self.x_i_pc_stim_avg_centered, axis=1)]
        self.y_data = [self.y_data[0] / np.max(self.y_data[0]), self.y_data[1] / np.max(self.y_data[1])]  # normalize

        plots.line_plot(self.x_data,self.y_data,colors=['blue', 'red'],linestyles=['--', '--'],labels=[r'$x_E$', r'$x_I$'],yticks=[0, 1],
                        y_max=1,y_min=0,ytickslabels=['0', '1'],xticks=[-90, -45, 0, 45, 90],
                        xticklabels=['', r'$-45^{\circ}$', r'$0^{\circ}$', r'$45^{\circ}$', ''], xlabel=r'$\Delta \theta$',
                        ylabel=r'Input to E (norm.)', figsize=[2.5, 2.0], title='Average E-I Tuning (PC)')

        # plot average E and I input to an I neuron
        # -------------------------------------
        self.x_data = [np.mean(self.theta_delta_pv, axis=0) * 180 / np.pi, np.mean(self.theta_delta_pv, axis=0) * 180 / np.pi]
        self.y_data = [np.mean(self.x_e_pv_stim_avg_centered, axis=1), np.mean(self.x_i_pv_stim_avg_centered, axis=1)]
        self.y_data = [self.y_data[0] / np.max(self.y_data[0]), self.y_data[1] / np.max(self.y_data[1])]  # normalize

        plots.line_plot(self.x_data,self.y_data, colors=['blue', 'red'],linestyles=['--', '--'],
            labels=[r'$x_E$', r'$x_I$'],
            # yticks=[0,0.25,0.5,0.75,1],
            yticks=[0, 1],
            # yticklabels= ['0','','','','1'],
            y_max=1,y_min=0,ytickslabels=['0', '1'],xticks=[-90, -45, 0, 45, 90],xticklabels=['', r'$-45^{\circ}$', r'$0^{\circ}$', r'$45^{\circ}$', ''],
            xlabel=r'$\Delta \theta$',ylabel=r'Input to I (norm.)',
            figsize=[2.5, 2.0],title='Average E-I Tuning (PV)')

        # plot average tuning curves - compare PV,PC firing rate tuning
        # -------------------------------------
        self.x_data = [np.mean(self.theta_delta_pc, axis=0) * 180 / np.pi, np.mean(self.theta_delta_pv, axis=0) * 180 / np.pi]
        self.y_data = [np.mean(self.y_pc_stim_avg_centered, axis=1), np.mean(self.y_pv_stim_avg_centered, axis=1)]
        self.y_data = [self.y_data[0] / np.max(self.y_data[0]), self.y_data[1] / np.max(self.y_data[1])]  # normalize
        plots.line_plot(
            self.x_data,
            self.y_data,
            colors=['blue', 'red', 'blue', 'red'],
            linestyles=['-', '-', '--', '--'],
            labels=[r'$y_E$', r'$y_I$'],
            # yticks=[0,0.25,0.5,0.75,1],
            yticks=[0, 1],
            # yticklabels= ['0','','','','1'],
            y_max=1,
            y_min=0,
            ytickslabels=['0', '1'],
            xticks=[-90, -45, 0, 45, 90],
            xticklabels=['', r'$-45^{\circ}$', r'$0^{\circ}$', r'$45^{\circ}$', ''],
            xlabel=r'$\Delta \theta$',
            ylabel=r'Firing rate (norm.)',
            figsize=[2.5, 2.0],title='Average Firing Rate Tuning (PC/PV')

        idx = self.sorted_pc_ids[int(self.N_pc / 2)]  # get neuron that is closest tuned to 90 degrees
        self.data = [self.x_i_pc_stim_avg[:, idx]]
        self.linestyles = ['-']
        self.colors = ['darkred']
        self.alphas = [1]
        for i in range(self.N_pv):
            self.data.append(self.M[idx, i] * self.y_pv_stim_avg[:, i])
            self.linestyles.append('-')
            self.colors.append('red')
            self.alphas.append(0.6)

        plots.line_plot(
            [self.theta_stim_avg * 180 / np.pi],
            self.data[1:],
            random_colors=[['darkred', 'red', 'lightred']],
            yticks=[],
            ytickslabels=[],
            xticks=[0, 45, 90, 135, 180],
            xticklabels=['', r'$45^{\circ}$', r'', r'$135^{\circ}$', ''],
            spine_visibility=[1, 0, 0, 0],
            xlabel=r'$\theta$',
            ylabel=r'Inhib. input',
            figsize=[2.5, 2.0],title='Inhibitory Input Composition (PC)')

        # plot connectivity matrices (sorted PV/PC)
        # -------------------------------------
        self.U_selector = tuple(np.meshgrid(self.sorted_pc_ids, self.sorted_pc_ids, indexing='ij'))
        self.Q_selector = tuple(np.meshgrid(self.sorted_pv_ids, self.sorted_pc_ids, indexing='ij'))
        self.P_selector = tuple(np.meshgrid(self.sorted_pv_ids, self.sorted_pv_ids, indexing='ij'))
        self.M_selector = tuple(np.meshgrid(self.sorted_pc_ids, self.sorted_pv_ids, indexing='ij'))

        self.U_sorted = self.U[self.U_selector]
        self.Q_sorted = self.Q[self.Q_selector]
        self.P_sorted = self.P[self.P_selector]
        self.M_sorted = self.M[self.M_selector]

        self.vmax = np.max([np.max(elem) for elem in [self.U, self.Q, self.P, self.M]])

        plots.connectivity_matrix([self.U_sorted, self.M_sorted, self.Q_sorted, self.P_sorted], xlabel=r'pre', ylabel=r'post')

        # plot connectivity kernels as function of tuning difference between pre- and postsynaptic neurons
        # -------------------------------------
        self.pc_tuning_peak *= 180 / np.pi
        self.pv_tuning_peak *= 180 / np.pi
        self.L4_tuning_peak *= 180 / np.pi

        # PC
        self.delta_theta_W = self.L4_tuning_peak - self.pc_tuning_peak[:, np.newaxis]
        self.delta_theta_W = ((self.delta_theta_W + 3 * 90) % 180) - 90  # correct for cyclic distance
        self.delta_theta_U = self.pc_tuning_peak - self.pc_tuning_peak[:, np.newaxis]
        self.delta_theta_U = ((self.delta_theta_U + 3 * 90) % 180) - 90  # correct for cyclic distance
        self.delta_theta_M = self.pv_tuning_peak - self.pc_tuning_peak[:, np.newaxis]
        self.delta_theta_M = ((self.delta_theta_M + 3 * 90) % 180) - 90  # correct for cyclic distance

        plots.scatter_plot(
            [self.delta_theta_W.flatten(), self.delta_theta_U.flatten(), self.delta_theta_M.flatten()],
            [self.W.flatten(), self.U.flatten(), self.M.flatten()],
            fit_gaussian=True,
            size=1,
            colors=['lightblue', 'blue', 'red'],
            labels=[r'$W_{EF}$', r'$W_{EE}$', r'$W_{EI}$'],
            xticks=[-90, -45, 0, 45, 90],
            xticklabels=['', r'$-45^{\circ}$', r'$0^{\circ}$', r'$45^{\circ}$', ''],
            xlabel=r'$\Delta \theta$',
            ylabel=r'syn. weight',
            legend=True, title='Connectivity Kernels (PC)')

        # PV
        self.delta_theta_K = self.L4_tuning_peak - self.pv_tuning_peak[:, np.newaxis]
        self.delta_theta_K = ((self.delta_theta_K + 3 * 90) % 180) - 90  # correct for cyclic distance
        self.delta_theta_Q = self.pc_tuning_peak - self.pv_tuning_peak[:, np.newaxis]
        self.delta_theta_Q = ((self.delta_theta_Q + 3 * 90) % 180) - 90  # correct for cyclic distance
        self.delta_theta_P = self.pv_tuning_peak - self.pv_tuning_peak[:, np.newaxis]
        self.delta_theta_P = ((self.delta_theta_P + 3 * 90) % 180) - 90  # correct for cyclic distance

        plots.scatter_plot(
            [self.delta_theta_K.flatten(), self.delta_theta_Q.flatten(), self.delta_theta_P.flatten()],
            [self.K.flatten(), self.Q.flatten(), self.P.flatten()],
            fit_gaussian=True,
            size=1,
            colors=['lightblue', 'blue', 'red'],
            labels=[r'$W_{IF}$', r'$W_{IE}$', r'$W_{II}$'],
            xticks=[-90, -45, 0, 45, 90],
            xticklabels=['', r'$-45^{\circ}$', r'$0^{\circ}$', r'$45^{\circ}$', ''],
            xlabel=r'$\Delta \theta$',
            ylabel=r'syn. weight',
            legend=True, title='Connectivity Kernels (PV)')

        #   plot recurrent kernels
        # -------------------------------------
        def gaussian(x, amp, mean, sigma):
            return amp * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

        self.x_data = [self.delta_theta_U.flatten(), self.delta_theta_M.flatten(), self.delta_theta_Q.flatten(), self.delta_theta_P.flatten()]
        self.y_data = [self.U.flatten(), -self.M.flatten(), self.Q.flatten(), -self.P.flatten()]
        self.y_data_fit = [[]] * len(self.y_data)
        for i in range(len(self.x_data)):
            self.amp_0 = np.sign(np.mean(self.y_data[i])) * np.max(abs(self.y_data[i]))
            if np.max(self.y_data[i]) == np.mean(self.y_data[i]):
                self.y_data_fit[i] = np.zeros((100,))  # if max equals min, everything is zero
            else:
                self.x_min = np.min(self.x_data[i])
                self.x_max = np.max(self.x_data[i])
                self.mean_0 = 0  # np.mean(x_data[i])
                self.std_0 = np.std(self.x_data[i]) / 2
                self.popt, self.pcov = optimize.curve_fit(f=gaussian, xdata=self.x_data[i], ydata=self.y_data[i], p0=[self.amp_0, self.mean_0, self.std_0], bounds=(
                (-abs(self.amp_0), self.x_min, (self.x_max - self.x_min) / 100), (abs(self.amp_0), self.x_max, (self.x_max - self.x_min))))
                self.x_samples = np.linspace(self.x_min, self.x_max, num=100)
                self.y_data_fit[i] = gaussian(self.x_samples, *self.popt)
                self.y_data_fit[i] /= np.max(abs(self.y_data_fit[i]))  # normalize to 1

        plots.line_plot(
            [self.x_samples],
            self.y_data_fit,
            colors=['blue', 'red', 'darkblue', 'darkred'],
            linestyles=['-', '-', (0, (2, 3)), (0, (2, 3))],
            labels=[r'$W_{EE}$', r'$W_{EI}$', r'$W_{IE}$', r'$W_{II}$'],
            xticks=[-90, -45, 0, 45, 90],
            xticklabels=['', r'$-45^{\circ}$', r'$0^{\circ}$', r'$45^{\circ}$', ''],
            yticks=[-1, 0, 1, 2],
            yticklabels=['-1', '0', '1', ''],
            y_max=1.5,
            xlabel=r'$\Delta \theta$',
            ylabel=r'Syn. weight (norm.)',
            figsize=[2.5, 2.0], title='Recurrent Connectivity Kernels')

        # plot feed forward weights before, during, and after training
        # -------------------------------------
        # self.W_hist = fn.extract_weight_hist(self.loaded_data[0], self.net_dic['W'], downSampleRatio=1)
        # self.K_hist = fn.extract_weight_hist(self.loaded_data[0], self.net_dic['K'], downSampleRatio=1)

        # PV ffwd weights
        # Plot K at end of simulation
        """
        self.K = self.K_hist[-1, :, :]
        self.K = np.hstack((self.K, self.K[:, [0]]))

        plots.line_plot([np.linspace(0, 180, self.K.shape[1])], [self.K.T], figsize=[4.2, 1.8],
                        random_colors=[['darkred', 'red', 'lightred']], spine_visibility=[1, 0, 0, 0], yticks=[],
                        yticklabels=[], xticks=[0, 45, 90, 135, 180],
                        xticklabels=['', '$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='', title='K After')

        # Plot K at beginning of simulation
        self.K = self.K_hist[5, :, :]
        self.K = np.hstack((self.K, self.K[:, [0]]))

        plots.line_plot([np.linspace(0, 180, self.K.shape[1])], [self.K.T], figsize=[4.2, 1.8],
                        random_colors=[['darkred', 'red', 'lightred']], spine_visibility=[1, 0, 0, 0], yticks=[],
                        yticklabels=[], xticks=[0, 45, 90, 135, 180],
                        xticklabels=['', '$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='',title='K Before')

        # Plot K at initialization of simulation
        self.K = self.K_hist[0, :, :]
        self.K = np.hstack((self.K, self.K[:, [0]]))

        plots.line_plot([np.linspace(0, 180, self.K.shape[1])], [self.K.T], figsize=[4.2, 1.8],
                        random_colors=[['darkred', 'red', 'lightred']], spine_visibility=[1, 0, 0, 0], yticks=[],
                        yticklabels=[], xticks=[0, 45, 90, 135, 180],
                        xticklabels=['', '$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='',title='K Init')

        # PC ffwd weights
        # Plot W at end of simulation
        self.W = self.W_hist[-1, :, :]
        self.W = np.hstack((self.W, self.W[:, [0]]))

        plots.line_plot([np.linspace(0, 180, self.W.shape[1])], [self.W.T], figsize=[4.2, 1.8],
                        random_colors=[['darkblue', 'blue', 'lightblue']], spine_visibility=[1, 0, 0, 0], yticks=[],
                        yticklabels=[], xticks=[0, 45, 90, 135, 180],
                        xticklabels=['', '$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='',title='W After')

        # Plot W at beginning of simulation
        self.W = self.W_hist[5, :, :]
        self.W = np.hstack((self.W, self.W[:, [0]]))

        plots.line_plot([np.linspace(0, 180, self.W.shape[1])], [self.W.T], figsize=[4.2, 1.8],
                        random_colors=[['darkblue', 'blue', 'lightblue']], spine_visibility=[1, 0, 0, 0], yticks=[],
                        yticklabels=[], xticks=[0, 45, 90, 135, 180],
                        xticklabels=['', '$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='',title='W Before')
        # Plot W at initialization of simulation
        self.W = self.W_hist[0, :, :]
        self.W = np.hstack((self.W, self.W[:, [0]]))

        plots.line_plot([np.linspace(0, 180, self.W.shape[1])], [self.W.T], figsize=[4.2, 1.8],
                        random_colors=[['darkblue', 'blue', 'lightblue']], spine_visibility=[1, 0, 0, 0], yticks=[],
                        yticklabels=[], xticks=[0, 45, 90, 135, 180],
                        xticklabels=['', '$45^\circ$', '', '$135^\circ$', ''], ylabel=r'', xlabel='',title='W Init')
        """
        plt.show()