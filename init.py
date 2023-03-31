
import datetime
import time
import shutil
import os
import importlib
import numpy as np

import libraries.loadsave as io
import libraries.functions as fn

#import config
#importlib.reload(config)  # make sure to load most recent config version
#from config import *  # import config file

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d--%H-%M-%S')

#-----------------------

save_path = "data/"+timestamp+"--"+short_description.replace(" ", "_")
save_path_figures = save_path+'/figures'
save_path_network = save_path+''
save_path_libraries = save_path+'/libraries'

for path in [save_path, save_path_figures, save_path_network, save_path_libraries]:
    fn.create_directory(path)

shutil.copy2("sim.py", save_path+"/sim.py")
shutil.copy2("init.py", save_path+"/init.py")
shutil.copy2("config.py", save_path+"/config.py")
shutil.copy2("libraries/inputs.py", save_path_libraries+"/inputs.py")
shutil.copy2("libraries/functions.py", save_path_libraries+"/functions.py")
shutil.copy2("libraries/loadsave.py", save_path_libraries+"/loadsave.py")
shutil.copy2("libraries/plots.py", save_path_libraries+"/plots.py")

##############################################################################

np.random.seed(random_seed)

###############################################################################
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
W      = abs(np.random.normal(2*std_W_EL, std_W_EL, (N_pc, N_L4)))         # (L4->PC)
K      = abs(np.random.normal(2*std_W_IL, std_W_IL, (N_pv, N_L4)))         # (L4->PV)
P      = abs(np.random.normal(2*std_W_II, std_W_II, (N_pv, N_pv)))         # (PV-|PV)
M      = abs(np.random.normal(2*std_W_EI, std_W_EI, (N_pc, N_pv)))         # (PV-|PC)
Q      = abs(np.random.normal(2*std_W_IE, std_W_IE, (N_pv, N_pc)))         # (PC->PV)
U      = abs(np.random.normal(2*std_W_EE, std_W_EE, (N_pc, N_pc)))         # (PC->PC)


###############################################################################
# history preallocation

network_states  = [[]] * (2 + int(T / accuracy_states))  # preallocate memory to track network state
count           = int(T / accuracy)                # number of entries in history arrays
t_hist          = np.zeros((count,))

# recorded from all neurons

# PC
x_pc_hist         = np.zeros((count, N_pc))
x_e_pc_hist       = np.zeros((count, N_pc))
x_i_pc_hist       = np.zeros((count, N_pc))
x_e_mean_pc_hist  = np.zeros((count, N_pc))
x_i_mean_pc_hist  = np.zeros((count, N_pc))
b_pc_hist         = np.zeros((count, N_pc))
a_pc_hist         = np.zeros((count, N_pc))
n_pc_hist         = np.zeros((count, N_pc))
W_EF_norms_hist   = np.zeros((count, N_pc))
W_EI_norms_hist   = np.zeros((count, N_pc))
W_EE_norms_hist   = np.zeros((count, N_pc))
y_pc_hist         = np.zeros((count, N_pc))
y_mean_pc_hist    = np.zeros((count, N_pc))
y_var_pc_hist     = np.zeros((count, N_pc))
y_0_pc_hist       = np.zeros((count, N_pc))

# PV
x_pv_hist         = np.zeros((count, N_pv))
x_e_pv_hist       = np.zeros((count, N_pv))
x_i_pv_hist       = np.zeros((count, N_pv))
x_e_mean_pv_hist  = np.zeros((count, N_pv))
x_i_mean_pv_hist  = np.zeros((count, N_pv))
b_pv_hist         = np.zeros((count, N_pv))
a_pv_hist         = np.zeros((count, N_pv))
n_pv_hist         = np.zeros((count, N_pv))
W_IF_norms_hist   = np.zeros((count, N_pv))
W_II_norms_hist   = np.zeros((count, N_pv))
W_IE_norms_hist   = np.zeros((count, N_pv))
y_pv_hist         = np.zeros((count, N_pv))
y_mean_pv_hist    = np.zeros((count, N_pv))
y_var_pv_hist     = np.zeros((count, N_pv))
y_0_pv_hist       = np.zeros((count, N_pv))

y_mean_L4_hist = np.zeros((count, N_L4))

# recorded from one neuron
W_hist = np.zeros((count, N_L4))  # (L4->PC)
K_hist = np.zeros((count, N_L4))  # (L4->PV)
P_hist = np.zeros((count, N_pv))  # (PV-|PV)
M_hist = np.zeros((count, N_pv))  # (PV-|PC)
Q_hist = np.zeros((count, N_pc))  # (PC->PV)
U_hist = np.zeros((count, N_pc))  # (PC->PC)

# recorded from all neurons
W_norm_hist = np.zeros((count, N_pc)) # (L4->PC)
K_norm_hist = np.zeros((count, N_pv)) # (L4->PV)
P_norm_hist = np.zeros((count, N_pv)) # (PV-|PV)
M_norm_hist = np.zeros((count, N_pc)) # (PV-|PC)
Q_norm_hist = np.zeros((count, N_pv)) # (PC->PV)
U_norm_hist = np.zeros((count, N_pc)) # (PC->PC)

# record variables every iteration during last 'T_fine' iterations

# PC
x_pc_hist_fine        = np.zeros((T_fine, N_pc))
x_e_pc_hist_fine      = np.zeros((T_fine, N_pc))
x_i_pc_hist_fine      = np.zeros((T_fine, N_pc))
x_e_mean_pc_hist_fine = np.zeros((T_fine, N_pc))
x_i_mean_pc_hist_fine = np.zeros((T_fine, N_pc))
b_pc_hist_fine        = np.zeros((T_fine, N_pc))
a_pc_hist_fine        = np.zeros((T_fine, N_pc))
n_pc_hist_fine        = np.zeros((T_fine, N_pc))
W_EF_norms_hist_fine  = np.zeros((T_fine, N_pc))
W_EI_norms_hist_fine  = np.zeros((T_fine, N_pc))
W_EE_norms_hist_fine  = np.zeros((T_fine, N_pc))
y_pc_hist_fine        = np.zeros((T_fine, N_pc))
y_mean_pc_hist_fine   = np.zeros((T_fine, N_pc))
y_var_pc_hist_fine    = np.zeros((T_fine, N_pc))
y_0_pc_hist_fine      = np.zeros((T_fine, N_pc))

# PV
x_pv_hist_fine        = np.zeros((T_fine, N_pv))
x_e_pv_hist_fine      = np.zeros((T_fine, N_pv))
x_i_pv_hist_fine      = np.zeros((T_fine, N_pv))
x_e_mean_pv_hist_fine = np.zeros((T_fine, N_pv))
x_i_mean_pv_hist_fine = np.zeros((T_fine, N_pv))
b_pv_hist_fine        = np.zeros((T_fine, N_pv))
a_pv_hist_fine        = np.zeros((T_fine, N_pv))
n_pv_hist_fine        = np.zeros((T_fine, N_pv))
W_IF_norms_hist_fine  = np.zeros((T_fine, N_pv))
W_II_norms_hist_fine  = np.zeros((T_fine, N_pv))
W_IE_norms_hist_fine  = np.zeros((T_fine, N_pv))
y_pv_hist_fine        = np.zeros((T_fine, N_pv))
y_mean_pv_hist_fine   = np.zeros((T_fine, N_pv))
y_var_pv_hist_fine    = np.zeros((T_fine, N_pv))
y_0_pv_hist_fine      = np.zeros((T_fine, N_pv))

y_L4_hist_fine      = np.zeros((T_fine, N_L4))
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
    d_theta_in = 180/N_L4
    d_theta_pc = 180/N_pc
    d_theta_pv = 180/N_pv
    theta_pc = np.arange(0, 180, d_theta_pc)  # tuning peaks of pc neurons
    theta_pv = np.arange(0, 180, d_theta_pv)  # tuning peaks of pv neurons
    theta_in = np.arange(0, 180, d_theta_in)  # tuning peaks of in neurons

    W = fn.gaussian(theta_pc, theta_in, input_sigma)  # weight connectivity kernels
    K = fn.gaussian(theta_pv, theta_in, input_sigma)
    U = fn.gaussian(theta_pc, theta_pc, sigma_ori)
    Q = fn.gaussian(theta_pv, theta_pc, sigma_ori)
    M = fn.gaussian(theta_pc, theta_pv, sigma_ori)
    P = fn.gaussian(theta_pv, theta_pv, sigma_ori)

norm_W = (np.sum(W**l_norm, axis=1))**(1/l_norm)  # EF
norm_K = (np.sum(K**l_norm, axis=1))**(1/l_norm)  # IF
norm_U = (np.sum(U**l_norm, axis=1))**(1/l_norm)  # EE
norm_M = (np.sum(M**l_norm, axis=1))**(1/l_norm)  # EI
norm_Q = (np.sum(Q**l_norm, axis=1))**(1/l_norm)  # IE
norm_P = (np.sum(P**l_norm, axis=1))**(1/l_norm)  # II


# after initialization, normalize input streams

# first establish feedforward and lateral weight norms as defined in config file (with small lateral E weights)
W[norm_W!=0,:] *= (W_EF_norms[norm_W!=0, np.newaxis] / norm_W[norm_W!=0, np.newaxis])
M[norm_M!=0,:] *= (W_EI_norms[norm_M!=0, np.newaxis] / norm_M[norm_M!=0, np.newaxis])
U[norm_U!=0,:] *= (W_EE_norms[norm_U!=0, np.newaxis] / norm_U[norm_U!=0, np.newaxis])

K[norm_K!=0,:] *= (W_IF_norms[norm_K!=0, np.newaxis] / norm_K[norm_K!=0, np.newaxis])
P[norm_P!=0,:] *= (W_II_norms[norm_P!=0, np.newaxis] / norm_P[norm_P!=0, np.newaxis])
Q[norm_Q!=0,:] *= (W_IE_norms[norm_Q!=0, np.newaxis] / norm_Q[norm_Q!=0, np.newaxis])

norm_W = (np.sum(W**l_norm, axis=1))**(1/l_norm)                                # EF
norm_K = (np.sum(K**l_norm, axis=1))**(1/l_norm)                                # IF
norm_U = (np.sum(U**l_norm, axis=1))**(1/l_norm)                                # EE
norm_M = (np.sum(M**l_norm, axis=1))**(1/l_norm)                                # EI
norm_Q = (np.sum(Q**l_norm, axis=1))**(1/l_norm)                                # IE
norm_P = (np.sum(P**l_norm, axis=1))**(1/l_norm)                                # II

# normalize all excitatory inputs (lateral and ffwd.) together
W_EE_norms = W_EE_norms + W_EF_norms
W_IE_norms = W_IE_norms + W_IF_norms

norm_WU = (norm_W**l_norm + norm_U**l_norm)**(1/l_norm)  # EE
norm_KQ = (norm_K**l_norm + norm_Q**l_norm)**(1/l_norm)  # IE

# normalization of exciatory PC input
W *= (W_EE_norms[:, np.newaxis] / norm_WU[:, np.newaxis])
U *= (W_EE_norms[:, np.newaxis] / norm_WU[:, np.newaxis])

# normalization of inhibitory PC input
M[norm_M!=0,:] *= (W_EI_norms[norm_M!=0, np.newaxis] / norm_M[norm_M!=0, np.newaxis])  # do not scale 0-norm input

# joint normalization of all exciatory PV input
K *= (W_IE_norms[:, np.newaxis] / norm_KQ[:, np.newaxis])
Q *= (W_IE_norms[:, np.newaxis] / norm_KQ[:, np.newaxis])

# normalization of inhibitory PV input
P[norm_P!=0,:] *= (W_II_norms[norm_P!=0, np.newaxis] / norm_P[norm_P!=0, np.newaxis]) # do not scale 0-norm input


# gather network state variables in a list for later saving
network = [W, K, P, M, Q, U,
           a_pc, b_pc, n_pc, W_EF_norms, W_EI_norms, W_EE_norms, y_mean_pc, y_var_pc, y_0_pc,
           a_pv, b_pv, n_pv, W_IF_norms, W_II_norms, W_IE_norms, y_mean_pv, y_var_pv, y_0_pv]

it = iter(np.arange(len(network))+1) # create iterator
net_dic = {"W":next(it), "K":next(it), "P":next(it), "M":next(it), "Q":next(it), "U":next(it),
           "a\_pc":next(it), "b\_pc":next(it), "n\_pc":next(it), "W\_EF\_norm":next(it), "W\_EI\_norm":next(it), "W\_EE\_norm":next(it), "y\_mean\_pc":next(it), "y\_var\_pc":next(it), "y\_0\_pc":next(it),
           "a\_pv":next(it), "b\_pv":next(it), "n\_pv":next(it), "W\_IF\_norm":next(it), "W\_II\_norm":next(it), "W\_IE\_norm":next(it), "y\_mean\_pv":next(it), "y\_var\_pv":next(it), "y\_0\_pv":next(it)}

hist = [x_pc_hist, x_e_pc_hist, x_i_pc_hist, x_e_mean_pc_hist, x_i_mean_pc_hist, b_pc_hist, a_pc_hist, n_pc_hist, W_EF_norms_hist, W_EI_norms_hist, W_EE_norms_hist, y_pc_hist, y_mean_pc_hist, y_var_pc_hist, y_0_pc_hist,
        x_pv_hist, x_e_pv_hist, x_i_pv_hist, x_e_mean_pv_hist, x_i_mean_pv_hist, b_pv_hist, a_pv_hist, n_pv_hist, W_IF_norms_hist, W_II_norms_hist, W_IE_norms_hist, y_pv_hist, y_mean_pv_hist, y_var_pv_hist, y_0_pv_hist,
        W_hist, K_hist, P_hist, M_hist, Q_hist, U_hist,
        W_norm_hist, K_norm_hist, P_norm_hist, M_norm_hist, Q_norm_hist, U_norm_hist]

hist_fine = [x_pc_hist_fine, x_e_pc_hist_fine, x_i_pc_hist_fine, x_e_mean_pc_hist_fine, x_i_mean_pc_hist_fine, b_pc_hist_fine, a_pc_hist_fine, n_pc_hist_fine, W_EF_norms_hist_fine, W_EI_norms_hist_fine, W_EE_norms_hist_fine, y_pc_hist_fine, y_mean_pc_hist_fine, y_var_pc_hist_fine, y_0_pc_hist_fine,
             x_pv_hist_fine, x_e_pv_hist_fine, x_i_pv_hist_fine, x_e_mean_pv_hist_fine, x_i_mean_pv_hist_fine, b_pv_hist_fine, a_pv_hist_fine, n_pv_hist_fine, W_IF_norms_hist_fine, W_II_norms_hist_fine, W_IE_norms_hist_fine, y_pv_hist_fine, y_mean_pv_hist_fine, y_var_pv_hist_fine, y_0_pv_hist_fine,
             y_L4_hist_fine,
             W_hist_fine, K_hist_fine, P_hist_fine, M_hist_fine, Q_hist_fine, U_hist_fine]
param = [e_x_pc, e_x_pv, e_y, e_w_EE, e_w_IE, e_w_EI, e_w_II,
         N_L4, N_pc, N_pv, input_amp, input_sigma]
